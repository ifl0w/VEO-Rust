use std::any::TypeId;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use crate::core::{Entity, EntityRef, Message};

#[macro_export]
macro_rules! filter {
    ( $( $x: ty ), * ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push(std::any::TypeId::of::<$x>());
            )*
            Filter {
                types: temp_vec,
                entities: vec![]
            }
        }
    }
}

#[derive(Clone)]
pub struct Filter {
    pub types: Vec<TypeId>,
    pub entities: Vec<EntityRef>,
}

impl Filter {
    pub fn update(&mut self, entities: &Vec<EntityRef>) {
        self.entities = entities
            .iter().cloned()
            .filter(|e| e.lock().ok().unwrap().match_filter(&self))
            .collect::<Vec<_>>();
    }
}

pub trait System {
    fn get_filter(&mut self) -> Vec<Filter> { vec![] }
    fn consume_messages(&mut self, _: &Vec<Message>) {}
    fn execute(&mut self, _: &Vec<Arc<Mutex<Filter>>>) {}
    fn get_messages(&mut self) -> Vec<Message> { vec![] }
}

pub struct SystemManager {
    pub systems: HashMap<TypeId, Arc<Mutex<dyn System>>>,
    pub filter: HashMap<TypeId, Vec<Arc<Mutex<Filter>>>>,
}

impl SystemManager {
    pub fn new() -> Self {
        SystemManager {
            systems: HashMap::new(),
            filter: HashMap::new(),
        }
    }

    pub fn add_system<T: 'static + System>(&mut self, sys: Arc<Mutex<T>>)
        where T: 'static + System {
        self.filter.insert(TypeId::of::<T>(), sys.lock().unwrap()
            .get_filter()
            .iter()
            .map(|f| Arc::new(Mutex::new(f.clone())))
            .collect());

        self.systems.insert(TypeId::of::<T>(), sys);
    }

    pub fn get_filter(&self, typeid: &TypeId) -> Option<&Vec<Arc<Mutex<Filter>>>> {
        self.filter.get(&typeid)
    }
}
