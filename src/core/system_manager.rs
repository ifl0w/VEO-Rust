use std::any::TypeId;
use std::fmt::Debug;

use crate::core::{Message, Entity};
use std::collections::HashMap;

pub trait System: Debug {
    fn consume_messages(&mut self, _: &Vec<Message>) { }
    fn execute(&mut self, _: &Vec<&Box<Entity>>) { }
    fn get_messages(&mut self) -> Vec<Message> { vec![] }
}

pub struct SystemManager {
    pub systems: HashMap<TypeId, Box<dyn System>>,
}

impl SystemManager {
    pub fn new() -> Self {
        SystemManager { systems: HashMap::new() }
    }

    pub fn add_system<T: 'static + System >(&mut self, sys_box: Box<T>)
        where T: 'static + System {
        self.systems.insert(TypeId::of::<T>(), sys_box);
    }
}
