use std::borrow::Borrow;
use std::convert::TryInto;
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufReader, Read};
use std::ops::Deref;
use std::path::Path;
use std::string::String;

pub struct ShaderCode {
    contents: String,
    path: String,

    binary_code: Vec<u8>,
    assembly_code: String,
}

impl ShaderCode {
    pub fn new(path: &str) -> Self {
        (ShaderCode {
            contents: String::new(),
            path: path.to_string(),

            binary_code: Vec::new(),
            assembly_code: String::new(),
        }).load_file(path)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        let mut sc = ShaderCode {
            contents: String::new(),
            path: String::new(),

            binary_code: Vec::new(),
            assembly_code: String::new(),
        };
        sc.contents = unsafe { String::from_utf8_unchecked(bytes) };

        sc
    }

    fn load_file(mut self, path: &str) -> Self {
        let file = File::open(path).expect(format!("Failed to open file ({}).", path).as_str());
        let mut buf_reader = BufReader::new(file);
        let mut contents = String::new();
        buf_reader.read_to_string(&mut contents).expect("Failed to read string from file.");

        self.contents = contents;

        self
    }

    pub fn compile(&mut self, shader_kind: shaderc::ShaderKind, entry_name: String) -> Option<(&[u8], &String)> {
        let mut compiler = shaderc::Compiler::new().unwrap();
        let mut options = shaderc::CompileOptions::new().unwrap();
        options.add_macro_definition("EP", Some(entry_name.as_str()));
        let binary_result = compiler.compile_into_spirv(
            self.contents.as_str(), shader_kind,
            self.path.as_str(),
            entry_name.as_str(), Some(&options));

        if binary_result.is_err() {
            println!("{}", binary_result.err().unwrap());

            return None;
        }

        let binary = binary_result.unwrap();
        self.binary_code = Vec::from(binary.as_binary_u8());

        let text_result = compiler.compile_into_spirv_assembly(
            self.contents.as_str(), shaderc::ShaderKind::Vertex,
            self.path.as_str(),
            entry_name.as_str(), Some(&options)).unwrap();

        self.assembly_code = text_result.as_text();


        Some((&self.binary_code, &self.assembly_code))
    }
}