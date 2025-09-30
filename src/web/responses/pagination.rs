//! Pagination utilities for list endpoints
//!
//! Provides reusable pagination helpers for applying limits, ordering,
//! and extracting cursor IDs from collections.

use crate::web::responses::constants::MAX_PAGINATION_LIMIT;

/// Pagination utilities for list endpoints
pub struct Paginator;

impl Paginator {
    /// Apply limit and check for more results
    ///
    /// Returns a tuple of (items, has_more) where items is truncated to the limit
    /// and has_more indicates if there were more items than the limit.
    ///
    /// # Arguments
    /// * `items` - The collection to paginate
    /// * `limit` - The maximum number of items to return
    ///
    /// # Example
    /// ```ignore
    /// let (items, has_more) = Paginator::paginate(items, 20);
    /// ```
    pub fn paginate<T>(mut items: Vec<T>, limit: i64) -> (Vec<T>, bool) {
        let limit = limit.min(MAX_PAGINATION_LIMIT) as usize;
        let has_more = items.len() > limit;

        if has_more {
            items.truncate(limit);
        }

        (items, has_more)
    }

    /// Reverse items if ascending order requested
    ///
    /// # Arguments
    /// * `items` - The collection to potentially reverse
    /// * `order` - Order string ("asc" or "desc")
    ///
    /// # Example
    /// ```ignore
    /// let items = Paginator::apply_order(items, "asc");
    /// ```
    pub fn apply_order<T>(mut items: Vec<T>, order: &str) -> Vec<T> {
        if order == "asc" {
            items.reverse();
        }
        items
    }

    /// Get first and last IDs from items using an ID extractor function
    ///
    /// # Arguments
    /// * `items` - The collection to extract IDs from
    /// * `id_extractor` - Function to extract ID from an item
    ///
    /// # Returns
    /// Tuple of (first_id, last_id) where both are Option<ID>
    ///
    /// # Example
    /// ```ignore
    /// let (first_id, last_id) = Paginator::get_cursor_ids(&items, |item| item.id);
    /// ```
    pub fn get_cursor_ids<T, ID, F>(items: &[T], id_extractor: F) -> (Option<ID>, Option<ID>)
    where
        F: Fn(&T) -> ID,
    {
        let first_id = items.first().map(&id_extractor);
        let last_id = items.last().map(&id_extractor);
        (first_id, last_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paginate_under_limit() {
        let items = vec![1, 2, 3];
        let (result, has_more) = Paginator::paginate(items, 5);
        assert_eq!(result, vec![1, 2, 3]);
        assert!(!has_more);
    }

    #[test]
    fn test_paginate_over_limit() {
        let items = vec![1, 2, 3, 4, 5];
        let (result, has_more) = Paginator::paginate(items, 3);
        assert_eq!(result, vec![1, 2, 3]);
        assert!(has_more);
    }

    #[test]
    fn test_paginate_respects_max_limit() {
        let items: Vec<i32> = (0..200).collect();
        let (result, has_more) = Paginator::paginate(items, 150); // Request 150 but max is 100
        assert_eq!(result.len(), MAX_PAGINATION_LIMIT as usize);
        assert!(has_more);
    }

    #[test]
    fn test_apply_order_desc() {
        let items = vec![1, 2, 3];
        let result = Paginator::apply_order(items, "desc");
        assert_eq!(result, vec![1, 2, 3]); // No change for desc
    }

    #[test]
    fn test_apply_order_asc() {
        let items = vec![1, 2, 3];
        let result = Paginator::apply_order(items, "asc");
        assert_eq!(result, vec![3, 2, 1]); // Reversed for asc
    }

    #[test]
    fn test_get_cursor_ids() {
        struct Item {
            id: i32,
        }
        let items = vec![Item { id: 1 }, Item { id: 2 }, Item { id: 3 }];
        let (first, last) = Paginator::get_cursor_ids(&items, |item| item.id);
        assert_eq!(first, Some(1));
        assert_eq!(last, Some(3));
    }

    #[test]
    fn test_get_cursor_ids_empty() {
        let items: Vec<i32> = vec![];
        let (first, last) = Paginator::get_cursor_ids(&items, |item| *item);
        assert_eq!(first, None);
        assert_eq!(last, None);
    }
}
