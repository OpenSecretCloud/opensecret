use std::io::{self, Write};

use zeroize::Zeroizing;

/// Zeroizing writer that never retains more than the configured JSON limit.
///
/// Callers may serialize one value or incrementally build a JSON sequence. A
/// write that would cross the limit fails before appending any of that chunk,
/// and the retained bytes are wiped when the buffer is dropped.
pub(crate) struct BoundedJsonBuffer {
    bytes: Zeroizing<Vec<u8>>,
    limit: usize,
    exceeded: bool,
}

impl BoundedJsonBuffer {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            bytes: Zeroizing::new(Vec::new()),
            limit,
            exceeded: false,
        }
    }

    pub(crate) fn into_bytes(mut self) -> Zeroizing<Vec<u8>> {
        Zeroizing::new(std::mem::take(&mut *self.bytes))
    }

    pub(crate) const fn exceeded(&self) -> bool {
        self.exceeded
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.bytes.len()
    }
}

impl Write for BoundedJsonBuffer {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let Some(next) = self.bytes.len().checked_add(buffer.len()) else {
            self.exceeded = true;
            return Err(io::Error::other("serialized JSON length overflow"));
        };
        if next > self.limit {
            self.exceeded = true;
            return Err(io::Error::other(
                "serialized JSON exceeds logical response limit",
            ));
        }
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeroize::Zeroize;

    #[test]
    fn exact_limit_succeeds_and_overflow_retains_only_bounded_bytes() {
        let mut exact = BoundedJsonBuffer::new(4);
        exact.write_all(b"1234").unwrap();
        assert_eq!(&*exact.into_bytes(), b"1234");

        let mut overflow = BoundedJsonBuffer::new(4);
        overflow.write_all(b"12").unwrap();
        assert!(overflow.write_all(b"345").is_err());
        assert!(overflow.exceeded());
        assert_eq!(overflow.len(), 2);
    }

    #[test]
    fn returned_bytes_remain_zeroizing() {
        fn assert_zeroize<T: Zeroize>() {}

        let mut buffer = BoundedJsonBuffer::new(2);
        buffer.write_all(b"[]").unwrap();
        let bytes = buffer.into_bytes();
        assert_zeroize::<Zeroizing<Vec<u8>>>();
        assert_eq!(&*bytes, b"[]");
    }
}
