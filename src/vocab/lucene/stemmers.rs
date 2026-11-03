//! Ports of Lucene 10.5.0 normalizers and light stemmers (`ArabicNormalizer`,
//! `ArabicStemmer`, `PersianNormalizer`, `PersianStemmer`, `IndicNormalizer`,
//! `HindiNormalizer`, `HindiStemmer`, `BengaliNormalizer`, `BengaliStemmer`,
//! `TeluguNormalizer`, `TeluguStemmer`, `IndonesianStemmer`,
//! `GermanNormalizationFilter`, `GermanLightStemmer`, `FrenchLightStemmer`,
//! `SpanishLightStemmer`), operating on UTF-16 buffers like the originals.

use super::indic_rules::{BENGALI_RULES, HINDI_RULES, TELUGU_RULES};
use super::util::{c, delete, delete_n, ends_with, starts_with, Term};

// ---------------------------------------------------------------- Arabic

pub(crate) fn arabic_normalize(s: &mut Term) {
    let mut i = 0;
    while i < s.len() {
        match s[i] {
            0x0622 | 0x0623 | 0x0625 => s[i] = 0x0627, // alef variants -> alef
            0x0649 => s[i] = 0x064A,                   // dotless yeh -> yeh
            0x0629 => s[i] = 0x0647,                   // teh marbuta -> heh
            // tatweel and harakat
            0x0640 | 0x064D | 0x064C | 0x064B | 0x064E | 0x064F | 0x0650 | 0x0651 | 0x0652 => {
                delete(s, i);
                continue;
            }
            _ => {}
        }
        i += 1;
    }
}

const AR_PREFIXES: &[&str] = &[
    "\u{0627}\u{0644}",
    "\u{0648}\u{0627}\u{0644}",
    "\u{0628}\u{0627}\u{0644}",
    "\u{0643}\u{0627}\u{0644}",
    "\u{0641}\u{0627}\u{0644}",
    "\u{0644}\u{0644}",
    "\u{0648}",
];

const AR_SUFFIXES: &[&str] = &[
    "\u{0647}\u{0627}",
    "\u{0627}\u{0646}",
    "\u{0627}\u{062A}",
    "\u{0648}\u{0646}",
    "\u{064A}\u{0646}",
    "\u{064A}\u{0647}",
    "\u{064A}\u{0629}",
    "\u{0647}",
    "\u{0629}",
    "\u{064A}",
];

pub(crate) fn arabic_stem(s: &mut Term) {
    // prefix (at most one)
    for p in AR_PREFIXES {
        let n = p.encode_utf16().count();
        let ok = if n == 1 && s.len() < 4 {
            false
        } else if s.len() < n + 2 {
            false
        } else {
            starts_with(s, p)
        };
        if ok {
            delete_n(s, 0, n);
            break;
        }
    }
    // suffixes (all that apply, in order)
    for suf in AR_SUFFIXES {
        let n = suf.encode_utf16().count();
        if s.len() >= n + 2 && ends_with(s, suf) {
            let l = s.len();
            delete_n(s, l - n, n);
        }
    }
}

// ---------------------------------------------------------------- Persian

pub(crate) fn persian_normalize(s: &mut Term) {
    let mut i = 0;
    while i < s.len() {
        match s[i] {
            0x06CC | 0x06D2 => s[i] = 0x064A, // farsi yeh, yeh barree -> yeh
            0x06A9 => s[i] = 0x0643,          // keheh -> kaf
            0x06C0 | 0x06C1 => s[i] = 0x0647, // heh yeh, heh goal -> heh
            0x0654 => {
                // hamza above
                delete(s, i);
                continue;
            }
            _ => {}
        }
        i += 1;
    }
}

const FA_SUFFIXES: &[&str] = &[
    "\u{0627}\u{062A}",
    "\u{0627}\u{0646}",
    "\u{062A}\u{0631}\u{064A}\u{0646}",
    "\u{062A}\u{0631}",
    "\u{064A}\u{064A}",
    "\u{064A}",
    "\u{0647}\u{0627}",
    "\u{200C}",
];

pub(crate) fn persian_stem(s: &mut Term) {
    for suf in FA_SUFFIXES {
        let n = suf.encode_utf16().count();
        if s.len() >= n + 2 && ends_with(s, suf) {
            let l = s.len();
            delete_n(s, l - n, n);
        }
    }
}

// ---------------------------------------------------------------- Indic

const DEVANAGARI: u32 = 1;
const BENGALI: u32 = 2;
const GURMUKHI: u32 = 4;
const GUJARATI: u32 = 8;
const ORIYA: u32 = 16;
const TAMIL: u32 = 32;
const TELUGU: u32 = 64;
const KANNADA: u32 = 128;
const MALAYALAM: u32 = 256;

/// (ch0, ch1, ch2 or -1 (0xFF = ZWJ), result, scripts), from `IndicNormalizer`
const DECOMPOSITIONS: &[(i32, i32, i32, i32, u32)] = &[
    (0x05, 0x3E, 0x45, 0x11, DEVANAGARI | GUJARATI),
    (0x05, 0x3E, 0x46, 0x12, DEVANAGARI),
    (0x05, 0x3E, 0x47, 0x13, DEVANAGARI | GUJARATI),
    (0x05, 0x3E, 0x48, 0x14, DEVANAGARI | GUJARATI),
    (
        0x05,
        0x3E,
        -1,
        0x06,
        DEVANAGARI | BENGALI | GURMUKHI | GUJARATI | ORIYA,
    ),
    (0x05, 0x45, -1, 0x72, DEVANAGARI),
    (0x05, 0x45, -1, 0x0D, GUJARATI),
    (0x05, 0x46, -1, 0x04, DEVANAGARI),
    (0x05, 0x47, -1, 0x0F, GUJARATI),
    (0x05, 0x48, -1, 0x10, GURMUKHI | GUJARATI),
    (0x05, 0x49, -1, 0x11, DEVANAGARI | GUJARATI),
    (0x05, 0x4A, -1, 0x12, DEVANAGARI),
    (0x05, 0x4B, -1, 0x13, DEVANAGARI | GUJARATI),
    (0x05, 0x4C, -1, 0x14, DEVANAGARI | GURMUKHI | GUJARATI),
    (0x06, 0x45, -1, 0x11, DEVANAGARI | GUJARATI),
    (0x06, 0x46, -1, 0x12, DEVANAGARI),
    (0x06, 0x47, -1, 0x13, DEVANAGARI | GUJARATI),
    (0x06, 0x48, -1, 0x14, DEVANAGARI | GUJARATI),
    (0x07, 0x57, -1, 0x08, MALAYALAM),
    (0x09, 0x41, -1, 0x0A, DEVANAGARI),
    (0x09, 0x57, -1, 0x0A, TAMIL | MALAYALAM),
    (0x0E, 0x46, -1, 0x10, MALAYALAM),
    (0x0F, 0x45, -1, 0x0D, DEVANAGARI),
    (0x0F, 0x46, -1, 0x0E, DEVANAGARI),
    (0x0F, 0x47, -1, 0x10, DEVANAGARI),
    (0x0F, 0x57, -1, 0x10, ORIYA),
    (0x12, 0x3E, -1, 0x13, MALAYALAM),
    (0x12, 0x4C, -1, 0x14, TELUGU | KANNADA),
    (0x12, 0x55, -1, 0x13, TELUGU),
    (0x12, 0x57, -1, 0x14, TAMIL | MALAYALAM),
    (0x13, 0x57, -1, 0x14, ORIYA),
    (0x15, 0x3C, -1, 0x58, DEVANAGARI),
    (0x16, 0x3C, -1, 0x59, DEVANAGARI | GURMUKHI),
    (0x17, 0x3C, -1, 0x5A, DEVANAGARI | GURMUKHI),
    (0x1C, 0x3C, -1, 0x5B, DEVANAGARI | GURMUKHI),
    (0x21, 0x3C, -1, 0x5C, DEVANAGARI | BENGALI | ORIYA),
    (0x22, 0x3C, -1, 0x5D, DEVANAGARI | BENGALI | ORIYA),
    (0x23, 0x4D, 0xFF, 0x7A, MALAYALAM),
    (0x24, 0x4D, 0xFF, 0x4E, BENGALI),
    (0x28, 0x3C, -1, 0x29, DEVANAGARI),
    (0x28, 0x4D, 0xFF, 0x7B, MALAYALAM),
    (0x2B, 0x3C, -1, 0x5E, DEVANAGARI | GURMUKHI),
    (0x2F, 0x3C, -1, 0x5F, DEVANAGARI | BENGALI),
    (0x2C, 0x41, 0x41, 0x0B, TELUGU),
    (0x30, 0x3C, -1, 0x31, DEVANAGARI),
    (0x30, 0x4D, 0xFF, 0x7C, MALAYALAM),
    (0x32, 0x4D, 0xFF, 0x7D, MALAYALAM),
    (0x33, 0x3C, -1, 0x34, DEVANAGARI),
    (0x33, 0x4D, 0xFF, 0x7E, MALAYALAM),
    (0x35, 0x41, -1, 0x2E, TELUGU),
    (0x3E, 0x45, -1, 0x49, DEVANAGARI | GUJARATI),
    (0x3E, 0x46, -1, 0x4A, DEVANAGARI),
    (0x3E, 0x47, -1, 0x4B, DEVANAGARI | GUJARATI),
    (0x3E, 0x48, -1, 0x4C, DEVANAGARI | GUJARATI),
    (0x3F, 0x55, -1, 0x40, KANNADA),
    (0x41, 0x41, -1, 0x42, GURMUKHI),
    (0x46, 0x3E, -1, 0x4A, TAMIL | MALAYALAM),
    (0x46, 0x42, 0x55, 0x4B, KANNADA),
    (0x46, 0x42, -1, 0x4A, KANNADA),
    (0x46, 0x46, -1, 0x48, MALAYALAM),
    (0x46, 0x55, -1, 0x47, TELUGU | KANNADA),
    (0x46, 0x56, -1, 0x48, TELUGU | KANNADA),
    (0x46, 0x57, -1, 0x4C, TAMIL | MALAYALAM),
    (0x47, 0x3E, -1, 0x4B, BENGALI | ORIYA | TAMIL | MALAYALAM),
    (0x47, 0x57, -1, 0x4C, BENGALI | ORIYA),
    (0x4A, 0x55, -1, 0x4B, KANNADA),
    (0x72, 0x3F, -1, 0x07, GURMUKHI),
    (0x72, 0x40, -1, 0x08, GURMUKHI),
    (0x72, 0x47, -1, 0x0F, GURMUKHI),
    (0x73, 0x41, -1, 0x09, GURMUKHI),
    (0x73, 0x42, -1, 0x0A, GURMUKHI),
    (0x73, 0x4B, -1, 0x13, GURMUKHI),
];

/// Unicode block of an Indic script (Devanagari..Malayalam): (flag, base)
fn indic_block(ch: u16) -> Option<(u32, u16)> {
    if (0x0900..0x0D80).contains(&ch) {
        let k = (ch - 0x0900) / 0x80;
        Some((1 << k, 0x0900 + k * 0x80))
    } else {
        None
    }
}

fn indic_decomp_mask(flag: u32, ch: i32) -> bool {
    DECOMPOSITIONS
        .iter()
        .any(|d| d.0 == ch && (d.4 & flag) != 0)
}

pub(crate) fn indic_normalize(s: &mut Term) {
    let mut i = 0;
    while i < s.len() {
        if let Some((flag, base)) = indic_block(s[i]) {
            let ch0 = (s[i] - base) as i32;
            if indic_decomp_mask(flag, ch0) {
                indic_compose(ch0, flag, base, s, i);
            }
        }
        i += 1;
    }
}

fn indic_compose(ch0: i32, flag: u32, base: u16, s: &mut Term, pos: usize) {
    if pos + 1 >= s.len() {
        return;
    }
    let block0 = indic_block(s[pos]);
    if indic_block(s[pos + 1]) != block0 {
        return;
    }
    let ch1 = s[pos + 1] as i32 - base as i32;
    let mut ch2 = -1;
    if pos + 2 < s.len() {
        ch2 = s[pos + 2] as i32 - base as i32;
        if s[pos + 2] == 0x200D {
            ch2 = 0xFF;
        } else if indic_block(s[pos + 2]) != block0 {
            ch2 = -1;
        }
    }
    for d in DECOMPOSITIONS {
        if d.0 == ch0 && (d.4 & flag) != 0 && d.1 == ch1 && (d.2 < 0 || d.2 == ch2) {
            s[pos] = base + d.3 as u16;
            delete(s, pos + 1);
            if d.2 >= 0 {
                delete(s, pos + 1);
            }
            return;
        }
    }
}

fn rule_stem(s: &mut Term, rules: &[(usize, &[&str])]) {
    for (min_len, suffixes) in rules {
        if s.len() > *min_len {
            if let Some(suf) = suffixes.iter().find(|suf| ends_with(s, suf)) {
                let n = suf.encode_utf16().count();
                s.truncate(s.len() - n);
                return;
            }
        }
    }
}

pub(crate) fn hindi_normalize(s: &mut Term) {
    let mut i = 0;
    while i < s.len() {
        match s[i] {
            // dead n -> bindu
            0x0928 => {
                if i + 1 < s.len() && s[i + 1] == 0x094D {
                    s[i] = 0x0902;
                    delete(s, i + 1);
                }
            }
            0x0901 => s[i] = 0x0902,
            // nukta, zwj/zwnj, virama: delete
            0x093C | 0x200D | 0x200C | 0x094D => {
                delete(s, i);
                continue;
            }
            0x0929 => s[i] = 0x0928,
            0x0931 => s[i] = 0x0930,
            0x0934 => s[i] = 0x0933,
            0x0958 => s[i] = 0x0915,
            0x0959 => s[i] = 0x0916,
            0x095A => s[i] = 0x0917,
            0x095B => s[i] = 0x091C,
            0x095C => s[i] = 0x0921,
            0x095D => s[i] = 0x0922,
            0x095E => s[i] = 0x092B,
            0x095F => s[i] = 0x092F,
            0x0945 | 0x0946 => s[i] = 0x0947,
            0x0949 | 0x094A => s[i] = 0x094B,
            0x090D | 0x090E => s[i] = 0x090F,
            0x0911 | 0x0912 => s[i] = 0x0913,
            0x0972 => s[i] = 0x0905,
            0x0906 => s[i] = 0x0905,
            0x0908 => s[i] = 0x0907,
            0x090A => s[i] = 0x0909,
            0x0960 => s[i] = 0x090B,
            0x0961 => s[i] = 0x090C,
            0x0910 => s[i] = 0x090F,
            0x0914 => s[i] = 0x0913,
            0x0940 => s[i] = 0x093F,
            0x0942 => s[i] = 0x0941,
            0x0944 => s[i] = 0x0943,
            0x0963 => s[i] = 0x0962,
            0x0948 => s[i] = 0x0947,
            0x094C => s[i] = 0x094B,
            _ => {}
        }
        i += 1;
    }
}

pub(crate) fn hindi_stem(s: &mut Term) {
    rule_stem(s, HINDI_RULES)
}

pub(crate) fn bengali_normalize(s: &mut Term) {
    // `i` is signed: some rules step back by two positions
    let mut i: isize = 0;
    while (i as usize) < s.len() {
        let u = i as usize;
        match s[u] {
            0x0981 => {
                delete(s, u);
                i -= 1;
            }
            0x09C0 => s[u] = 0x09BF,
            0x09C2 => s[u] = 0x09C1,
            0x0995 => {
                if u + 2 < s.len() && s[u + 1] == 0x09CD && s[u + 2] == 0x09BF {
                    if u == 0 {
                        s[u] = 0x0996;
                        delete(s, u + 2);
                        delete(s, u + 1);
                    } else {
                        s[u + 1] = 0x0996;
                        delete(s, u + 2);
                    }
                }
            }
            0x0999 => s[u] = 0x0982,
            0x09AF => {
                if i - 2 == 0 && s[u - 1] == 0x09CD {
                    s[u - 1] = 0x09C7;
                    if u + 1 < s.len() && s[u + 1] == 0x09BE {
                        delete(s, u + 1);
                    }
                    delete(s, u);
                    i -= 1;
                } else if i >= 1 && s[u - 1] == 0x09CD {
                    delete(s, u);
                    delete(s, u - 1);
                    i -= 2;
                }
            }
            0x09AC => {
                if !((i >= 1 && s[u - 1] != 0x09CD) || i == 0) {
                    if i - 2 == 0 {
                        delete(s, u);
                        delete(s, u - 1);
                        i -= 2;
                    } else if i - 5 >= 0 && s[u - 3] == 0x09CD {
                        delete(s, u);
                        delete(s, u - 1);
                        i -= 2;
                    } else if i - 2 >= 0 {
                        s[u - 1] = s[u - 2];
                        delete(s, u);
                        i -= 1;
                    }
                }
            }
            0x0983 => {
                if u == s.len() - 1 {
                    if s.len() <= 3 {
                        s[u] = 0x09B9;
                    } else {
                        delete(s, u);
                    }
                } else {
                    s[u] = s[u + 1];
                }
            }
            0x09B6 | 0x09B7 => s[u] = 0x09B8,
            0x09A3 => s[u] = 0x09A8,
            0x09DC | 0x09DD => s[u] = 0x09B0,
            0x09CE => s[u] = 0x09A4,
            _ => {}
        }
        i += 1;
    }
}

pub(crate) fn bengali_stem(s: &mut Term) {
    rule_stem(s, BENGALI_RULES)
}

pub(crate) fn telugu_normalize(s: &mut Term) {
    let mut i = 0;
    while i < s.len() {
        match s[i] {
            0x0C00 | 0x0C01 => s[i] = 0x0C02,
            0x0C03 | 0x200D | 0x200C => {
                delete(s, i);
                continue;
            }
            0x0C14 => s[i] = 0x0C13,
            0x0C10 => s[i] = 0x0C0F,
            0x0C06 => s[i] = 0x0C05,
            0x0C08 => s[i] = 0x0C07,
            0x0C0A => s[i] = 0x0C09,
            0x0C40 => s[i] = 0x0C3F,
            0x0C42 => s[i] = 0x0C41,
            0x0C47 => s[i] = 0x0C46,
            0x0C4B => s[i] = 0x0C4A,
            0x0C46 => {
                if i + 1 < s.len() && s[i + 1] == 0x0C56 {
                    s[i] = 0x0C48;
                    delete(s, i + 1);
                }
            }
            0x0C12 => {
                if i + 1 < s.len() && s[i + 1] == 0x0C55 {
                    s[i] = 0x0C13;
                    delete(s, i + 1);
                } else if i + 1 < s.len() && s[i + 1] == 0x0C4C {
                    s[i] = 0x0C14;
                    delete(s, i + 1);
                }
            }
            _ => {}
        }
        i += 1;
    }
}

pub(crate) fn telugu_stem(s: &mut Term) {
    rule_stem(s, TELUGU_RULES)
}

// ---------------------------------------------------------------- Indonesian

const REMOVED_KE: u32 = 1;
const REMOVED_PENG: u32 = 2;
const REMOVED_DI: u32 = 4;
const REMOVED_MENG: u32 = 8;
const REMOVED_TER: u32 = 16;
const REMOVED_BER: u32 = 32;
const REMOVED_PE: u32 = 64;

fn id_vowel(ch: u16) -> bool {
    matches!(ch, 0x61 | 0x65 | 0x69 | 0x6F | 0x75) // a e i o u
}

struct Indonesian {
    syllables: i32,
    flags: u32,
}

impl Indonesian {
    fn remove_particle(&mut self, s: &mut Term) {
        if ends_with(s, "kah") || ends_with(s, "lah") || ends_with(s, "pun") {
            self.syllables -= 1;
            s.truncate(s.len() - 3);
        }
    }

    fn remove_possessive_pronoun(&mut self, s: &mut Term) {
        if ends_with(s, "ku") || ends_with(s, "mu") {
            self.syllables -= 1;
            s.truncate(s.len() - 2);
        } else if ends_with(s, "nya") {
            self.syllables -= 1;
            s.truncate(s.len() - 3);
        }
    }

    /// Removes `n` leading chars, recording `flag`
    fn strip(&mut self, s: &mut Term, n: usize, flag: u32) -> bool {
        self.flags |= flag;
        self.syllables -= 1;
        delete_n(s, 0, n);
        true
    }

    fn remove_first_order_prefix(&mut self, s: &mut Term) -> bool {
        let l = s.len();
        if starts_with(s, "meng") {
            return self.strip(s, 4, REMOVED_MENG);
        }
        if starts_with(s, "meny") && l > 4 && id_vowel(s[4]) {
            s[3] = c('s');
            return self.strip(s, 3, REMOVED_MENG);
        }
        if starts_with(s, "men") || starts_with(s, "mem") {
            return self.strip(s, 3, REMOVED_MENG);
        }
        if starts_with(s, "me") {
            return self.strip(s, 2, REMOVED_MENG);
        }
        if starts_with(s, "peng") {
            return self.strip(s, 4, REMOVED_PENG);
        }
        if starts_with(s, "peny") && l > 4 && id_vowel(s[4]) {
            s[3] = c('s');
            return self.strip(s, 3, REMOVED_PENG);
        }
        if starts_with(s, "peny") {
            return self.strip(s, 4, REMOVED_PENG);
        }
        if starts_with(s, "pen") && l > 3 && id_vowel(s[3]) {
            s[2] = c('t');
            return self.strip(s, 2, REMOVED_PENG);
        }
        if starts_with(s, "pen") || starts_with(s, "pem") {
            return self.strip(s, 3, REMOVED_PENG);
        }
        if starts_with(s, "di") {
            return self.strip(s, 2, REMOVED_DI);
        }
        if starts_with(s, "ter") {
            return self.strip(s, 3, REMOVED_TER);
        }
        if starts_with(s, "ke") {
            return self.strip(s, 2, REMOVED_KE);
        }
        false
    }

    fn remove_second_order_prefix(&mut self, s: &mut Term) -> bool {
        let l = s.len();
        if starts_with(s, "ber") {
            return self.strip(s, 3, REMOVED_BER);
        }
        if l == 7 && starts_with(s, "belajar") {
            return self.strip(s, 3, REMOVED_BER);
        }
        if starts_with(s, "be") && l > 4 && !id_vowel(s[2]) && s[3] == c('e') && s[4] == c('r') {
            return self.strip(s, 2, REMOVED_BER);
        }
        if starts_with(s, "per") {
            return self.strip(s, 3, 0);
        }
        if l == 7 && starts_with(s, "pelajar") {
            return self.strip(s, 3, 0);
        }
        if starts_with(s, "pe") {
            return self.strip(s, 2, REMOVED_PE);
        }
        false
    }

    fn remove_suffix(&mut self, s: &mut Term) -> bool {
        let f = self.flags;
        if ends_with(s, "kan") && f & (REMOVED_KE | REMOVED_PENG | REMOVED_PE) == 0 {
            self.syllables -= 1;
            s.truncate(s.len() - 3);
            return true;
        }
        if ends_with(s, "an") && f & (REMOVED_DI | REMOVED_MENG | REMOVED_TER) == 0 {
            self.syllables -= 1;
            s.truncate(s.len() - 2);
            return true;
        }
        if ends_with(s, "i")
            && !ends_with(s, "si")
            && f & (REMOVED_BER | REMOVED_KE | REMOVED_PENG) == 0
        {
            self.syllables -= 1;
            s.truncate(s.len() - 1);
            return true;
        }
        false
    }
}

/// `IndonesianStemmer.stem(..., stemDerivational=true)`
pub(crate) fn indonesian_stem(s: &mut Term) {
    let mut st = Indonesian {
        syllables: s.iter().filter(|&&ch| id_vowel(ch)).count() as i32,
        flags: 0,
    };
    if st.syllables > 2 {
        st.remove_particle(s);
    }
    if st.syllables > 2 {
        st.remove_possessive_pronoun(s);
    }
    // stemDerivational
    let fired = st.syllables > 2 && st.remove_first_order_prefix(s);
    if fired {
        let fired = st.syllables > 2 && st.remove_suffix(s);
        if fired && st.syllables > 2 {
            st.remove_second_order_prefix(s);
        }
    } else {
        if st.syllables > 2 {
            st.remove_second_order_prefix(s);
        }
        if st.syllables > 2 {
            st.remove_suffix(s);
        }
    }
}

// ---------------------------------------------------------------- German

/// `GermanNormalizationFilter`
pub(crate) fn german_normalize(s: &mut Term) {
    const N: u8 = 0;
    const V: u8 = 1;
    const U: u8 = 2;
    let mut state = N;
    let mut i = 0;
    while i < s.len() {
        match s[i] {
            0x61 | 0x6F => state = U,                       // a o
            0x75 => state = if state == N { U } else { V }, // u
            0x65 => {
                // e
                if state == U {
                    delete(s, i);
                    state = V;
                    continue;
                }
                state = V;
            }
            0x69 | 0x71 | 0x79 => state = V, // i q y
            0xE4 => {
                s[i] = c('a');
                state = V;
            }
            0xF6 => {
                s[i] = c('o');
                state = V;
            }
            0xFC => {
                s[i] = c('u');
                state = V;
            }
            0xDF => {
                // ß -> ss
                s[i] = c('s');
                i += 1;
                s.insert(i, c('s'));
                state = N;
            }
            _ => state = N,
        }
        i += 1;
    }
}

fn st_ending(ch: u16) -> bool {
    matches!(
        char::from_u32(ch as u32),
        Some('b' | 'd' | 'f' | 'g' | 'h' | 'k' | 'l' | 'm' | 'n' | 't')
    )
}

/// `GermanLightStemmer`
pub(crate) fn german_light_stem(s: &mut Term) {
    for ch in s.iter_mut() {
        *ch = match *ch {
            0xE4 | 0xE0 | 0xE1 | 0xE2 => c('a'),
            0xF6 | 0xF2 | 0xF3 | 0xF4 => c('o'),
            0xEF | 0xEC | 0xED | 0xEE => c('i'),
            0xFC | 0xF9 | 0xFA | 0xFB => c('u'),
            other => other,
        }
    }
    // step 1
    let l = s.len();
    let new_len = if l > 5 && s[l - 3] == c('e') && s[l - 2] == c('r') && s[l - 1] == c('n') {
        l - 3
    } else if l > 4 && s[l - 2] == c('e') && matches!(s[l - 1], 0x6D | 0x6E | 0x72 | 0x73) {
        l - 2
    } else if l > 3 && s[l - 1] == c('e') {
        l - 1
    } else if l > 3 && s[l - 1] == c('s') && st_ending(s[l - 2]) {
        l - 1
    } else {
        l
    };
    s.truncate(new_len);
    // step 2
    let l = s.len();
    let new_len = if l > 5 && s[l - 3] == c('e') && s[l - 2] == c('s') && s[l - 1] == c('t') {
        l - 3
    } else if l > 4 && s[l - 2] == c('e') && (s[l - 1] == c('r') || s[l - 1] == c('n')) {
        l - 2
    } else if l > 4 && s[l - 2] == c('s') && s[l - 1] == c('t') && st_ending(s[l - 3]) {
        l - 2
    } else {
        l
    };
    s.truncate(new_len);
}

// ---------------------------------------------------------------- Spanish

/// `SpanishLightStemmer`
pub(crate) fn spanish_light_stem(s: &mut Term) {
    let l = s.len();
    if l < 5 {
        return;
    }
    for ch in s.iter_mut() {
        *ch = match *ch {
            0xE0 | 0xE1 | 0xE2 | 0xE4 => c('a'),
            0xF2 | 0xF3 | 0xF4 | 0xF6 => c('o'),
            0xE8 | 0xE9 | 0xEA | 0xEB => c('e'),
            0xF9 | 0xFA | 0xFB | 0xFC => c('u'),
            0xEC | 0xED | 0xEE | 0xEF => c('i'),
            other => other,
        }
    }
    match s[l - 1] {
        0x6F | 0x61 | 0x65 => s.truncate(l - 1), // o a e
        0x73 => {
            // s
            if s[l - 2] == c('e') && s[l - 3] == c('s') && s[l - 4] == c('e') {
                s.truncate(l - 2);
            } else if s[l - 2] == c('e') && s[l - 3] == c('c') {
                s[l - 3] = c('z');
                s.truncate(l - 2);
            } else if matches!(s[l - 2], 0x6F | 0x61 | 0x65) {
                s.truncate(l - 2);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------- French

const FRENCH_ARTICLES: &[&str] = &[
    "l", "m", "t", "qu", "n", "s", "j", "d", "c", "jusqu", "quoiqu", "lorsqu", "puisqu",
];

/// `ElisionFilter` with French's default articles (case-insensitive)
pub(crate) fn french_elision(token: &str) -> String {
    if let Some(index) = token.find(['\'', '\u{2019}']) {
        let prefix: String = super::util::lowercase(&token[..index]);
        if FRENCH_ARTICLES.contains(&prefix.as_str()) {
            let skip = token[index..].chars().next().unwrap().len_utf8();
            return token[index + skip..].to_string();
        }
    }
    token.to_string()
}

/// `Character.isLetter(char)` (surrogates are not letters), approximated as
/// alphabetic minus the common combining-mark blocks and letter numbers --
/// only used on French terms
fn is_letter(ch: u16) -> bool {
    char::from_u32(ch as u32).is_some_and(|c| {
        c.is_alphabetic()
            && !matches!(c as u32,
                0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF
                | 0xFE20..=0xFE2F | 0x2160..=0x2188)
    })
}

/// `FrenchLightStemmer`
pub(crate) fn french_light_stem(s: &mut Term) {
    let mut len = s.len();
    macro_rules! set {
        ($i:expr, $ch:expr) => {
            s[$i] = c($ch)
        };
    }
    macro_rules! ends {
        ($suf:expr) => {
            ends_with(&s[..len], $suf)
        };
    }
    macro_rules! done {
        ($l:expr) => {{
            let l = $l;
            let l = french_norm(s, l);
            s.truncate(l);
            return;
        }};
    }

    if len > 5 && s[len - 1] == c('x') {
        if s[len - 3] == c('a') && s[len - 2] == c('u') && s[len - 4] != c('e') {
            set!(len - 2, 'l');
        }
        len -= 1;
    }
    if len > 3 && s[len - 1] == c('x') {
        len -= 1;
    }
    if len > 3 && s[len - 1] == c('s') {
        len -= 1;
    }
    if len > 9 && ends!("issement") {
        len -= 6;
        set!(len - 1, 'r');
        done!(len);
    }
    if len > 8 && ends!("issant") {
        len -= 4;
        set!(len - 1, 'r');
        done!(len);
    }
    if len > 6 && ends!("ement") {
        len -= 4;
        if len > 3 && ends!("ive") {
            len -= 1;
            set!(len - 1, 'f');
        }
        done!(len);
    }
    if len > 11 && ends!("ficatrice") {
        len -= 5;
        set!(len - 2, 'e');
        set!(len - 1, 'r');
        done!(len);
    }
    if len > 10 && ends!("ficateur") {
        len -= 4;
        set!(len - 2, 'e');
        set!(len - 1, 'r');
        done!(len);
    }
    if len > 9 && ends!("catrice") {
        len -= 3;
        set!(len - 4, 'q');
        set!(len - 3, 'u');
        set!(len - 2, 'e');
        done!(len);
    }
    if len > 8 && ends!("cateur") {
        len -= 2;
        set!(len - 4, 'q');
        set!(len - 3, 'u');
        set!(len - 2, 'e');
        set!(len - 1, 'r');
        done!(len);
    }
    if len > 8 && ends!("atrice") {
        len -= 4;
        set!(len - 2, 'e');
        set!(len - 1, 'r');
        done!(len);
    }
    if len > 7 && ends!("ateur") {
        len -= 3;
        set!(len - 2, 'e');
        set!(len - 1, 'r');
        done!(len);
    }
    if len > 6 && ends!("trice") {
        len -= 1;
        set!(len - 3, 'e');
        set!(len - 2, 'u');
        set!(len - 1, 'r');
    }
    if len > 5 && ends!("ième") {
        done!(len - 4);
    }
    if len > 7 && ends!("teuse") {
        len -= 2;
        set!(len - 1, 'r');
        done!(len);
    }
    if len > 6 && ends!("teur") {
        len -= 1;
        set!(len - 1, 'r');
        done!(len);
    }
    if len > 5 && ends!("euse") {
        done!(len - 2);
    }
    if len > 8 && ends!("ère") {
        len -= 1;
        set!(len - 2, 'e');
        done!(len);
    }
    if len > 7 && ends!("ive") {
        len -= 1;
        set!(len - 1, 'f');
        done!(len);
    }
    if len > 4 && (ends!("folle") || ends!("molle")) {
        len -= 2;
        set!(len - 1, 'u');
        done!(len);
    }
    if len > 9 && ends!("nnelle") {
        done!(len - 5);
    }
    if len > 9 && ends!("nnel") {
        done!(len - 3);
    }
    if len > 4 && ends!("ète") {
        len -= 1;
        set!(len - 2, 'e');
    }
    if len > 8 && ends!("ique") {
        len -= 4;
    }
    if len > 8 && ends!("esse") {
        done!(len - 3);
    }
    if len > 7 && ends!("inage") {
        done!(len - 3);
    }
    if len > 9 && ends!("isation") {
        len -= 7;
        if len > 5 && ends!("ual") {
            set!(len - 2, 'e');
        }
        done!(len);
    }
    if len > 9 && ends!("isateur") {
        done!(len - 7);
    }
    if len > 8 && ends!("ation") {
        done!(len - 5);
    }
    if len > 8 && ends!("ition") {
        done!(len - 5);
    }
    done!(len);
}

/// `FrenchLightStemmer.norm`: works on `s[..len]`, returns the new length
fn french_norm(s: &mut Term, mut len: usize) -> usize {
    if len > 4 {
        for ch in s[..len].iter_mut() {
            *ch = match *ch {
                0xE0 | 0xE1 | 0xE2 => c('a'),
                0xF4 => c('o'),
                0xE8 | 0xE9 | 0xEA => c('e'),
                0xF9 | 0xFB => c('u'),
                0xEE => c('i'),
                0xE7 => c('c'),
                other => other,
            }
        }
        let mut ch = s[0];
        let mut i = 1;
        while i < len {
            if s[i] == ch && is_letter(ch) {
                s.remove(i);
                len -= 1;
            } else {
                ch = s[i];
                i += 1;
            }
        }
    }
    if len > 4 && ends_with(&s[..len], "ie") {
        len -= 2;
    }
    if len > 4 {
        if s[len - 1] == c('r') {
            len -= 1;
        }
        if s[len - 1] == c('e') {
            len -= 1;
        }
        if s[len - 1] == c('e') {
            len -= 1;
        }
        if s[len - 1] == s[len - 2] && is_letter(s[len - 1]) {
            len -= 1;
        }
    }
    len
}

#[cfg(test)]
mod tests {
    use super::super::util::{from_term, to_term};
    use super::*;

    fn apply(f: fn(&mut Term), s: &str) -> String {
        let mut t = to_term(s);
        f(&mut t);
        from_term(&t)
    }

    #[test]
    // Expected values: Lucene 10.5.0 analyzers
    fn test_german() {
        assert_eq!(apply(german_normalize, "straße"), "strasse");
        assert_eq!(apply(german_normalize, "häuser"), "hauser");
        assert_eq!(apply(german_light_stem, "kinder"), "kind");
    }

    #[test]
    fn test_french() {
        assert_eq!(apply(french_light_stem, "chevaux"), "cheval");
        assert_eq!(french_elision("l'école"), "école");
        assert_eq!(french_elision("L'école"), "école");
        assert_eq!(french_elision("aujourd'hui"), "aujourd'hui");
    }

    #[test]
    fn test_spanish() {
        assert_eq!(apply(spanish_light_stem, "niños"), "niñ");
        assert_eq!(apply(spanish_light_stem, "luces"), "luz");
    }

    #[test]
    fn test_indonesian() {
        assert_eq!(apply(indonesian_stem, "bermain"), "main");
        assert_eq!(apply(indonesian_stem, "sekolah"), "seko");
    }
}
