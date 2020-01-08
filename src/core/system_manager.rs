use std::any::TypeId;
use std::collections::HashMap;

use crate::core::{Entity, Message, EntityRef};
use std::sync::{Arc, Mutex};

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
    fn execute(&mut self, _: &Vec<EntityRef>) {}
    fn get_messages(&mut self) -> Vec<Message> { vec![] }
}

pub struct SystemManager {
    pub systems: HashMap<TypeId, Arc<Mutex<dyn System>>>,
}

impl SystemManager {
    pub fn new() -> Self {
        SystemManager { systems: HashMap::new() }
    }

    pub fn add_system<T: 'static + System>(&mut self, sys: Arc<Mutex<T>>)
        where T: 'static + System {
        self.systems.insert(TypeId::of::<T>(), sys);
    }
}
