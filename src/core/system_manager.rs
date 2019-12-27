pub trait System {
    fn execute(&self);
}

pub struct SystemManager {
    systems: Vec<Box<dyn System>>,
}

impl SystemManager {
    pub fn new() -> Self {
        SystemManager { systems: vec![] }
    }

    pub fn add_system(&mut self, s: Box<dyn System>) {
        println!("adding system");
        self.systems.push(s);
    }
}
