//! UTF-16 term buffers, mirroring Lucene's `char[]` term attributes so that
//! ported filters keep Lucene's exact length semantics (lengths are counted
//! in UTF-16 code units, like Java's `char`).

/// A term buffer (UTF-16 code units)
pub(crate) type Term = Vec<u16>;

pub(crate) fn to_term(s: &str) -> Term {
    s.encode_utf16().collect()
}

pub(crate) fn from_term(t: &[u16]) -> String {
    String::from_utf16_lossy(t)
}

/// Number of UTF-16 code units of `s`
pub(crate) fn len16(s: &str) -> usize {
    s.encode_utf16().count()
}

/// `StemmerUtil.endsWith`
pub(crate) fn ends_with(s: &[u16], suffix: &str) -> bool {
    let n = len16(suffix);
    n <= s.len() && s[s.len() - n..].iter().copied().eq(suffix.encode_utf16())
}

/// `StemmerUtil.startsWith`
pub(crate) fn starts_with(s: &[u16], prefix: &str) -> bool {
    let n = len16(prefix);
    n <= s.len() && s[..n].iter().copied().eq(prefix.encode_utf16())
}

/// `StemmerUtil.delete`
pub(crate) fn delete(s: &mut Term, pos: usize) {
    s.remove(pos);
}

/// `StemmerUtil.deleteN`
pub(crate) fn delete_n(s: &mut Term, pos: usize, n: usize) {
    s.drain(pos..pos + n);
}

pub(crate) const fn c(ch: char) -> u16 {
    ch as u16
}

/// Code points of a UTF-16 buffer (unpaired surrogates kept as is)
pub(crate) fn code_points(s: &[u16]) -> Vec<u32> {
    char::decode_utf16(s.iter().copied())
        .map(|r| {
            r.map(|c| c as u32)
                .unwrap_or_else(|e| e.unpaired_surrogate() as u32)
        })
        .collect()
}

/// Java's `Character.toLowerCase(int)` (simple case mapping): Rust's full
/// lowercase mapping only differs for U+0130 (to "i̇" instead of "i")
pub(crate) fn lower_char(ch: char) -> char {
    if ch == '\u{130}' {
        return 'i';
    }
    let mut it = ch.to_lowercase();
    match (it.next(), it.next()) {
        (Some(l), None) => l,
        _ => ch,
    }
}

/// `LowerCaseFilter`
pub(crate) fn lowercase(s: &str) -> String {
    s.chars().map(lower_char).collect()
}

/// Zero of each block of Unicode decimal digits (`Character.isDigit`)
/// outside Basic Latin, as of Java 21 (generated with `Character`)
const DIGIT_ZEROS: &[u32] = &[
    0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66, 0xDE6,
    0xE50, 0xED0, 0xF20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90, 0x1B50,
    0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10,
    0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0, 0x11650,
    0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x11F50, 0x16A60, 0x16AC0,
    0x16B50, 0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E4F0, 0x1E950,
    0x1FBF0,
];

fn decimal_value(ch: u32) -> Option<u32> {
    if ch <= 0x7F {
        return None;
    }
    let i = DIGIT_ZEROS.partition_point(|&z| z <= ch);
    if i == 0 {
        return None;
    }
    let zero = DIGIT_ZEROS[i - 1];
    (ch - zero < 10).then_some(ch - zero)
}

/// Non-ASCII decimal digit (`Character.isDigit`)
pub(crate) fn is_decimal_digit(ch: char) -> bool {
    decimal_value(ch as u32).is_some()
}

/// `DecimalDigitFilter`: non-ASCII decimal digits to ASCII
pub(crate) fn decimal_digits(s: &str) -> String {
    s.chars()
        .map(|ch| match decimal_value(ch as u32) {
            Some(v) => char::from_digit(v, 10).unwrap(),
            None => ch,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decimal_digits() {
        assert_eq!(decimal_digits("١٢٣ ०९ x9"), "123 09 x9");
        assert_eq!(decimal_digits("\u{1D7CE}\u{1D7D7}"), "09");
    }

    #[test]
    fn test_lowercase() {
        assert_eq!(lowercase("İSTANBUL ΣΑΣ"), "istanbul σασ");
    }

    #[test]
    fn test_ends_starts() {
        let t = to_term("abcdef");
        assert!(ends_with(&t, "def"));
        assert!(!ends_with(&t, "xdef"));
        assert!(starts_with(&t, "abc"));
        assert!(!starts_with(&t, "abcdefg"));
    }
}
