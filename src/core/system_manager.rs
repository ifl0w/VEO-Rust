use std::any::TypeId;
use std::collections::HashMap;

use crate::core::{Entity, Message};

#[macro_export]
macro_rules! filter {
    ( $( $x: ty ), * ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push(std::any::TypeId::of::<$x>());
            )*
            Some(Filter{
                types: temp_vec
            })
        }
    }
}

pub struct Filter {
    pub types: Vec<TypeId>
}

pub trait System {
    fn get_filter(&mut self) -> Option<Filter> { None }
    fn consume_messages(&mut self, _: &Vec<Message>) {}
    fn execute(&mut self, _: &Vec<&Box<Entity>>) {}
    fn get_messages(&mut self) -> Vec<Message> { vec![] }
}

pub struct SystemManager {
    pub systems: HashMap<TypeId, Box<dyn System>>,
}

impl SystemManager {
    pub fn new() -> Self {
        SystemManager { systems: HashMap::new() }
    }

    pub fn add_system<T: 'static + System>(&mut self, sys_box: Box<T>)
        where T: 'static + System {
        self.systems.insert(TypeId::of::<T>(), sys_box);
    }
}
