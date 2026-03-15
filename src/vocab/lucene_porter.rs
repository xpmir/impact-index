//! Lucene-compatible Porter stemmer.
//!
//! Direct port of `org.apache.lucene.analysis.en.PorterStemmer` from Apache Lucene.
//! Translates the Java code faithfully to match Lucene's exact stemming behavior
//! character by character.
//!
//! Based on:
//!   Porter, 1980, "An algorithm for suffix stripping", Program, Vol. 14, no. 3, pp 130-137.
//!   See also <https://snowballstem.org/algorithms/porter/stemmer.html>
//!
//! Includes Bug 1 and Bug 2 fixes from the original Lucene implementation.
//!
//! Note: Java uses signed `int` for all indices. We use `i32` to match that
//! behavior faithfully, since `j` can become negative (e.g. Bug 2 fix).

/// Lucene-compatible Porter stemmer.
/// Direct port of org.apache.lucene.analysis.en.PorterStemmer.
pub struct LucenePorterStemmer {
    b: Vec<char>, // buffer
    i: i32,       // offset into b (length of word in buffer)
    j: i32,       // general offset
    k: i32,       // offset to end of string
    k0: i32,      // start offset
    dirty: bool,
}

impl LucenePorterStemmer {
    pub fn new() -> Self {
        LucenePorterStemmer {
            b: Vec::with_capacity(50),
            i: 0,
            j: 0,
            k: 0,
            k0: 0,
            dirty: false,
        }
    }

    /// Stem a word. Returns the stemmed form.
    pub fn stem(&mut self, word: &str) -> String {
        self.reset();
        for ch in word.chars() {
            self.add(ch);
        }
        if self.stem_impl(0) {
            self.to_result()
        } else {
            word.to_string()
        }
    }

    fn reset(&mut self) {
        self.i = 0;
        self.dirty = false;
        self.b.clear();
    }

    fn add(&mut self, ch: char) {
        let idx = self.i as usize;
        if self.b.len() <= idx {
            self.b.push(ch);
        } else {
            self.b[idx] = ch;
        }
        self.i += 1;
    }

    fn to_result(&self) -> String {
        self.b[..self.i as usize].iter().collect()
    }

    /// cons(i) is true <=> b[i] is a consonant.
    fn cons(&self, i: i32) -> bool {
        match self.b[i as usize] {
            'a' | 'e' | 'i' | 'o' | 'u' => false,
            'y' => {
                if i == self.k0 {
                    true
                } else {
                    !self.cons(i - 1)
                }
            }
            _ => true,
        }
    }

    /// m() measures the number of consonant sequences between k0 and j.
    fn m(&self) -> i32 {
        let mut n: i32 = 0;
        let mut i = self.k0;
        loop {
            if i > self.j {
                return n;
            }
            if !self.cons(i) {
                break;
            }
            i += 1;
        }
        i += 1;
        loop {
            loop {
                if i > self.j {
                    return n;
                }
                if self.cons(i) {
                    break;
                }
                i += 1;
            }
            i += 1;
            n += 1;
            loop {
                if i > self.j {
                    return n;
                }
                if !self.cons(i) {
                    break;
                }
                i += 1;
            }
            i += 1;
        }
    }

    /// vowelinstem() is true <=> k0,...j contains a vowel
    fn vowelinstem(&self) -> bool {
        let mut i = self.k0;
        while i <= self.j {
            if !self.cons(i) {
                return true;
            }
            i += 1;
        }
        false
    }

    /// doublec(j) is true <=> j,(j-1) contain a double consonant.
    fn doublec(&self, j: i32) -> bool {
        if j < self.k0 + 1 {
            return false;
        }
        if self.b[j as usize] != self.b[(j - 1) as usize] {
            return false;
        }
        self.cons(j)
    }

    /// cvc(i) is true <=> i-2,i-1,i has the form consonant - vowel - consonant
    /// and also if the second c is not w,x or y.
    fn cvc(&self, i: i32) -> bool {
        if i < self.k0 + 2 || !self.cons(i) || self.cons(i - 1) || !self.cons(i - 2) {
            return false;
        }
        let ch = self.b[i as usize];
        if ch == 'w' || ch == 'x' || ch == 'y' {
            return false;
        }
        true
    }

    fn ends(&mut self, s: &str) -> bool {
        let l = s.len() as i32;
        let o = self.k - l + 1;
        if o < self.k0 {
            return false;
        }
        for (idx, ch) in s.chars().enumerate() {
            if self.b[(o + idx as i32) as usize] != ch {
                return false;
            }
        }
        self.j = self.k - l;
        true
    }

    /// setto(s) sets (j+1),...k to the characters in the string s, readjusting k.
    fn setto(&mut self, s: &str) {
        let l = s.len() as i32;
        let o = self.j + 1;
        for (idx, ch) in s.chars().enumerate() {
            self.b[(o + idx as i32) as usize] = ch;
        }
        self.k = self.j + l;
        self.dirty = true;
    }

    /// r(s) is used further down.
    fn r(&mut self, s: &str) {
        if self.m() > 0 {
            self.setto(s);
        }
    }

    /// step1() gets rid of plurals and -ed or -ing.
    fn step1(&mut self) {
        if self.b[self.k as usize] == 's' {
            if self.ends("sses") {
                self.k -= 2;
            } else if self.ends("ies") {
                self.setto("i");
            } else if self.b[(self.k - 1) as usize] != 's' {
                self.k -= 1;
            }
        }
        if self.ends("eed") {
            if self.m() > 0 {
                self.k -= 1;
            }
        } else if (self.ends("ed") || self.ends("ing")) && self.vowelinstem() {
            self.k = self.j;
            if self.ends("at") {
                self.setto("ate");
            } else if self.ends("bl") {
                self.setto("ble");
            } else if self.ends("iz") {
                self.setto("ize");
            } else if self.doublec(self.k) {
                let ch = self.b[self.k as usize];
                self.k -= 1;
                if ch == 'l' || ch == 's' || ch == 'z' {
                    self.k += 1;
                }
            } else if self.m() == 1 && self.cvc(self.k) {
                self.setto("e");
            }
        }
    }

    /// step2() turns terminal y to i when there is another vowel in the stem.
    fn step2(&mut self) {
        if self.ends("y") && self.vowelinstem() {
            self.b[self.k as usize] = 'i';
            self.dirty = true;
        }
    }

    /// step3() maps double suffices to single ones.
    fn step3(&mut self) {
        if self.k == self.k0 {
            return; /* For Bug 1 */
        }
        match self.b[(self.k - 1) as usize] {
            'a' => {
                if self.ends("ational") {
                    self.r("ate");
                    return;
                }
                if self.ends("tional") {
                    self.r("tion");
                    return;
                }
            }
            'c' => {
                if self.ends("enci") {
                    self.r("ence");
                    return;
                }
                if self.ends("anci") {
                    self.r("ance");
                    return;
                }
            }
            'e' => {
                if self.ends("izer") {
                    self.r("ize");
                    return;
                }
            }
            'l' => {
                if self.ends("bli") {
                    self.r("ble");
                    return;
                }
                if self.ends("alli") {
                    self.r("al");
                    return;
                }
                if self.ends("entli") {
                    self.r("ent");
                    return;
                }
                if self.ends("eli") {
                    self.r("e");
                    return;
                }
                if self.ends("ousli") {
                    self.r("ous");
                    return;
                }
            }
            'o' => {
                if self.ends("ization") {
                    self.r("ize");
                    return;
                }
                if self.ends("ation") {
                    self.r("ate");
                    return;
                }
                if self.ends("ator") {
                    self.r("ate");
                    return;
                }
            }
            's' => {
                if self.ends("alism") {
                    self.r("al");
                    return;
                }
                if self.ends("iveness") {
                    self.r("ive");
                    return;
                }
                if self.ends("fulness") {
                    self.r("ful");
                    return;
                }
                if self.ends("ousness") {
                    self.r("ous");
                    return;
                }
            }
            't' => {
                if self.ends("aliti") {
                    self.r("al");
                    return;
                }
                if self.ends("iviti") {
                    self.r("ive");
                    return;
                }
                if self.ends("biliti") {
                    self.r("ble");
                    return;
                }
            }
            'g' => {
                if self.ends("logi") {
                    self.r("log");
                    return;
                }
            }
            _ => {}
        }
    }

    /// step4() deals with -ic-, -full, -ness etc.
    fn step4(&mut self) {
        match self.b[self.k as usize] {
            'e' => {
                if self.ends("icate") {
                    self.r("ic");
                    return;
                }
                if self.ends("ative") {
                    self.r("");
                    return;
                }
                if self.ends("alize") {
                    self.r("al");
                    return;
                }
            }
            'i' => {
                if self.ends("iciti") {
                    self.r("ic");
                    return;
                }
            }
            'l' => {
                if self.ends("ical") {
                    self.r("ic");
                    return;
                }
                if self.ends("ful") {
                    self.r("");
                    return;
                }
            }
            's' => {
                if self.ends("ness") {
                    self.r("");
                    return;
                }
            }
            _ => {}
        }
    }

    /// step5() takes off -ant, -ence etc., in context <c>vcvc<v>.
    fn step5(&mut self) {
        if self.k == self.k0 {
            return; /* for Bug 1 */
        }
        match self.b[(self.k - 1) as usize] {
            'a' => {
                if !self.ends("al") {
                    return;
                }
            }
            'c' => {
                if !self.ends("ance") && !self.ends("ence") {
                    return;
                }
            }
            'e' => {
                if !self.ends("er") {
                    return;
                }
            }
            'i' => {
                if !self.ends("ic") {
                    return;
                }
            }
            'l' => {
                if !self.ends("able") && !self.ends("ible") {
                    return;
                }
            }
            'n' => {
                if !self.ends("ant")
                    && !self.ends("ement")
                    && !self.ends("ment")
                    && !self.ends("ent")
                {
                    return;
                }
            }
            'o' => {
                // "ion" requires j >= 0 and b[j] == 's' or 't' (Bug 2 fix)
                let ion_match = self.ends("ion")
                    && self.j >= 0
                    && (self.b[self.j as usize] == 's' || self.b[self.j as usize] == 't');
                if !ion_match && !self.ends("ou") {
                    return;
                }
            }
            's' => {
                if !self.ends("ism") {
                    return;
                }
            }
            't' => {
                if !self.ends("ate") && !self.ends("iti") {
                    return;
                }
            }
            'u' => {
                if !self.ends("ous") {
                    return;
                }
            }
            'v' => {
                if !self.ends("ive") {
                    return;
                }
            }
            'z' => {
                if !self.ends("ize") {
                    return;
                }
            }
            _ => {
                return;
            }
        }
        if self.m() > 1 {
            self.k = self.j;
        }
    }

    /// step6() removes a final -e if m() > 1.
    fn step6(&mut self) {
        self.j = self.k;
        if self.b[self.k as usize] == 'e' {
            let a = self.m();
            if a > 1 || (a == 1 && !self.cvc(self.k - 1)) {
                self.k -= 1;
            }
        }
        if self.b[self.k as usize] == 'l' && self.doublec(self.k) && self.m() > 1 {
            self.k -= 1;
        }
    }

    fn stem_impl(&mut self, i0: i32) -> bool {
        self.k = self.i - 1;
        self.k0 = i0;
        if self.k > self.k0 + 1 {
            self.step1();
            self.step2();
            self.step3();
            self.step4();
            self.step5();
            self.step6();
        }
        // Also, a word is considered dirty if we lopped off letters
        // Thanks to Ifigenia Vairelles for pointing this out.
        if self.i != self.k + 1 {
            self.dirty = true;
        }
        self.i = self.k + 1;
        self.dirty
    }
}

impl super::stemmer::Stemmer for LucenePorterStemmer {
    fn stem(&self, word: &str) -> String {
        // The Stemmer trait requires &self, so we create a fresh instance per call.
        let mut s = LucenePorterStemmer::new();
        LucenePorterStemmer::stem(&mut s, word)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_stemming() {
        let mut stemmer = LucenePorterStemmer::new();
        assert_eq!(stemmer.stem("caresses"), "caress");
        assert_eq!(stemmer.stem("ponies"), "poni");
        assert_eq!(stemmer.stem("ties"), "ti");
        assert_eq!(stemmer.stem("caress"), "caress");
        assert_eq!(stemmer.stem("cats"), "cat");
    }

    #[test]
    fn test_ed_ing() {
        let mut stemmer = LucenePorterStemmer::new();
        assert_eq!(stemmer.stem("feed"), "feed");
        assert_eq!(stemmer.stem("agreed"), "agre");
        assert_eq!(stemmer.stem("disabled"), "disabl");
        assert_eq!(stemmer.stem("matting"), "mat");
        assert_eq!(stemmer.stem("mating"), "mate");
        assert_eq!(stemmer.stem("meeting"), "meet");
        assert_eq!(stemmer.stem("milling"), "mill");
        assert_eq!(stemmer.stem("messing"), "mess");
        assert_eq!(stemmer.stem("meetings"), "meet");
    }

    #[test]
    fn test_step2() {
        let mut stemmer = LucenePorterStemmer::new();
        assert_eq!(stemmer.stem("happy"), "happi");
    }

    #[test]
    fn test_step3() {
        let mut stemmer = LucenePorterStemmer::new();
        assert_eq!(stemmer.stem("relational"), "relat");
        assert_eq!(stemmer.stem("conditional"), "condit");
        assert_eq!(stemmer.stem("rational"), "ration");
    }

    #[test]
    fn test_short_words() {
        let mut stemmer = LucenePorterStemmer::new();
        // Short words should pass through
        assert_eq!(stemmer.stem("a"), "a");
        assert_eq!(stemmer.stem("be"), "be");
        assert_eq!(stemmer.stem("the"), "the");
    }

    #[test]
    fn test_stemmer_trait() {
        use super::super::stemmer::Stemmer;
        let stemmer = LucenePorterStemmer::new();
        assert_eq!(Stemmer::stem(&stemmer, "caresses"), "caress");
        assert_eq!(Stemmer::stem(&stemmer, "ponies"), "poni");
    }

    #[test]
    fn test_step5_ion() {
        let mut stemmer = LucenePorterStemmer::new();
        // "ion" should only be stripped when preceded by 's' or 't'
        assert_eq!(stemmer.stem("adoption"), "adopt");
        assert_eq!(stemmer.stem("communion"), "communion");
    }

    #[test]
    fn test_various_words() {
        let mut stemmer = LucenePorterStemmer::new();
        assert_eq!(stemmer.stem("generalization"), "gener");
        assert_eq!(stemmer.stem("oscillators"), "oscil");
        assert_eq!(stemmer.stem("communism"), "commun");
    }

    /// Validate against known Lucene stems from Pyserini's analyzer
    #[test]
    fn test_lucene_compatibility() {
        let mut s = LucenePorterStemmer::new();
        // High-frequency words where Snowball (Porter2) differs from Lucene
        let cases = vec![
            // -ment words: Lucene preserves, Snowball preserves, old porter-stemmer over-stems
            ("document", "document"),
            ("documents", "document"),
            ("element", "element"),
            ("supplement", "supplement"),
            ("argument", "argument"),
            ("instrument", "instrument"),
            ("movement", "movement"),
            ("statement", "statement"),
            ("agreement", "agreement"),
            ("settlement", "settlement"),
            // -y -> -i: Lucene converts, Snowball doesn't
            ("day", "dai"),
            ("days", "dai"),
            ("way", "wai"),
            ("play", "plai"),
            ("say", "sai"),
            ("bay", "bai"),
            ("money", "monei"),
            ("key", "kei"),
            ("turkey", "turkei"),
            ("birthday", "birthdai"),
            ("journey", "journei"),
            // -us -> stripped: Lucene strips, Snowball doesn't
            ("virus", "viru"),
            ("bonus", "bonu"),
            ("campus", "campu"),
            ("sinus", "sinu"),
            ("venus", "venu"),
            // -ly: Lucene keeps -li, Snowball strips further
            ("commonly", "commonli"),
            ("quickly", "quickli"),
            ("directly", "directli"),
            ("properly", "properli"),
            // use/used/uses: Lucene -> us, Snowball -> use
            ("use", "us"),
            ("used", "us"),
            ("uses", "us"),
            ("using", "us"),
            // community/communication: Lucene -> commun
            ("community", "commun"),
            ("communication", "commun"),
            // generation/general: Lucene -> gener
            ("generation", "gener"),
            ("generously", "gener"),
            // organization: Lucene -> organ
            ("organization", "organ"),
            // university: Lucene -> univers
            ("university", "univers"),
            // Common stems that should match both
            ("running", "run"),
            ("jumps", "jump"),
            ("easily", "easili"),
        ];
        for (word, expected) in &cases {
            let got = s.stem(word);
            assert_eq!(
                &got, expected,
                "stem('{}') = '{}', expected '{}'",
                word, got, expected
            );
        }
    }
}
