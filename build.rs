fn main() {
    // Only the Python extension needs the extension-module link arguments
    if std::env::var_os("CARGO_FEATURE_PYTHON").is_some() {
        pyo3_build_config::add_extension_module_link_args();
    }
}
