use std::{borrow::Borrow, collections::HashMap, hash::Hash, time::Duration};
use tokio::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CapacityError;

struct Entry<V> {
    value: V,
    expires_at: Instant,
}

/// A fixed-capacity map whose entries expire after a period of inactivity.
///
/// Callers are expected to put this behind a lock. Capacity checks and inserts
/// then happen atomically, so concurrent requests cannot exceed the configured
/// memory bound.
pub(crate) struct BoundedTtlMap<K, V> {
    entries: HashMap<K, Entry<V>>,
    capacity: usize,
    idle_ttl: Duration,
}

impl<K, V> BoundedTtlMap<K, V>
where
    K: Eq + Hash,
{
    pub(crate) fn new(capacity: usize, idle_ttl: Duration) -> Self {
        assert!(capacity > 0, "bounded map capacity must be non-zero");
        assert!(!idle_ttl.is_zero(), "bounded map TTL must be non-zero");

        Self {
            entries: HashMap::new(),
            capacity,
            idle_ttl,
        }
    }

    pub(crate) fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, CapacityError> {
        self.try_insert_at(key, value, Instant::now())
    }

    pub(crate) fn try_insert_at(
        &mut self,
        key: K,
        value: V,
        now: Instant,
    ) -> Result<Option<V>, CapacityError> {
        self.prune_expired_at(now);

        if self.entries.len() >= self.capacity && !self.entries.contains_key(&key) {
            return Err(CapacityError);
        }

        Ok(self
            .entries
            .insert(
                key,
                Entry {
                    value,
                    expires_at: now + self.idle_ttl,
                },
            )
            .map(|entry| entry.value))
    }

    pub(crate) fn remove_live<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.remove_live_at(key, Instant::now())
    }

    fn remove_live_at<Q>(&mut self, key: &Q, now: Instant) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let entry = self.entries.remove(key)?;
        (entry.expires_at > now).then_some(entry.value)
    }

    pub(crate) fn get_cloned_and_touch_at(&mut self, key: &K, now: Instant) -> Option<V>
    where
        V: Clone,
    {
        if self.entries.get(key)?.expires_at <= now {
            self.entries.remove(key);
            return None;
        }

        let entry = self.entries.get_mut(key)?;
        entry.expires_at = now + self.idle_ttl;
        Some(entry.value.clone())
    }

    fn prune_expired_at(&mut self, now: Instant) {
        self.entries.retain(|_, entry| entry.expires_at > now);
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use tokio::sync::{Barrier, RwLock};

    const TTL: Duration = Duration::from_secs(10);

    #[test]
    fn capacity_rejection_preserves_live_entries() {
        let now = Instant::now();
        let mut map = BoundedTtlMap::new(2, TTL);

        assert_eq!(map.try_insert_at("one", 1, now), Ok(None));
        assert_eq!(map.try_insert_at("two", 2, now), Ok(None));
        assert_eq!(map.try_insert_at("three", 3, now), Err(CapacityError));

        assert_eq!(map.len(), 2);
        assert_eq!(map.get_cloned_and_touch_at(&"one", now), Some(1));
        assert_eq!(map.get_cloned_and_touch_at(&"two", now), Some(2));
    }

    #[test]
    fn insert_prunes_expired_entries_without_removing_live_entries() {
        let now = Instant::now();
        let mut map = BoundedTtlMap::new(2, TTL);

        map.try_insert_at("expired", 1, now).unwrap();
        map.try_insert_at("live", 2, now + Duration::from_secs(5))
            .unwrap();

        assert_eq!(
            map.try_insert_at("new", 3, now + TTL),
            Ok(None),
            "the expired slot should be reclaimed"
        );
        assert_eq!(map.len(), 2);
        assert_eq!(map.get_cloned_and_touch_at(&"live", now + TTL), Some(2));
        assert_eq!(map.get_cloned_and_touch_at(&"expired", now + TTL), None);
    }

    #[test]
    fn removal_is_one_time_and_rejects_expired_entries() {
        let now = Instant::now();
        let mut map = BoundedTtlMap::new(2, TTL);

        map.try_insert_at("once", 1, now).unwrap();
        assert_eq!(map.remove_live_at(&"once", now), Some(1));
        assert_eq!(map.remove_live_at(&"once", now), None);

        map.try_insert_at("expired", 2, now).unwrap();
        assert_eq!(map.remove_live_at(&"expired", now + TTL), None);
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn successful_access_extends_idle_expiry() {
        let now = Instant::now();
        let mut map = BoundedTtlMap::new(1, TTL);
        map.try_insert_at("session", 7, now).unwrap();

        assert_eq!(
            map.get_cloned_and_touch_at(&"session", now + Duration::from_secs(5)),
            Some(7)
        );
        assert_eq!(
            map.get_cloned_and_touch_at(&"session", now + Duration::from_secs(14)),
            Some(7),
            "touching the session should keep an active request alive"
        );
        assert_eq!(
            map.get_cloned_and_touch_at(&"session", now + Duration::from_secs(24)),
            None
        );
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn expired_lookup_drops_the_value_immediately() {
        struct DropSpy(Arc<AtomicUsize>);

        impl Drop for DropSpy {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let now = Instant::now();
        let drops = Arc::new(AtomicUsize::new(0));
        let mut map = BoundedTtlMap::new(1, TTL);
        map.try_insert_at("session", Arc::new(DropSpy(Arc::clone(&drops))), now)
            .unwrap();

        assert!(map.get_cloned_and_touch_at(&"session", now + TTL).is_none());
        assert_eq!(map.len(), 0);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn concurrent_inserts_cannot_exceed_capacity() {
        const CAPACITY: usize = 8;
        const REQUESTS: usize = 64;

        let map = Arc::new(RwLock::new(BoundedTtlMap::new(CAPACITY, TTL)));
        let barrier = Arc::new(Barrier::new(REQUESTS));
        let now = Instant::now();
        let mut tasks = Vec::with_capacity(REQUESTS);

        for key in 0..REQUESTS {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            tasks.push(tokio::spawn(async move {
                barrier.wait().await;
                map.write().await.try_insert_at(key, key, now).is_ok()
            }));
        }

        let mut successful_inserts = 0;
        for task in tasks {
            successful_inserts += usize::from(task.await.unwrap());
        }

        assert_eq!(successful_inserts, CAPACITY);
        assert_eq!(map.read().await.len(), CAPACITY);
    }
}
