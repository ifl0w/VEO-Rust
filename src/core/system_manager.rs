use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

//use winit::Event;

use crate::core::{EntityRef, Message};
use winit::event::Event;

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

    pub fn add(&mut self, e: EntityRef) {
        if e.lock().unwrap().match_filter(&self) {
            self.entities.push(e);
        }
    }

    pub fn remove(&mut self, e: EntityRef) {
        let index = self.entities.iter().position(|x| {
            x.lock().unwrap().id == e.lock().unwrap().id
        });

        match index {
            Some(idx) => {
                self.entities.remove(idx);
            },
            None => ()
        }
    }
}

pub trait System {
    fn get_filter(&mut self) -> Vec<Filter> { vec![] }
    fn handle_input(&mut self, _event: &Event<()>) {}
    fn consume_messages(&mut self, _: &Vec<Message>) {}
    fn execute(&mut self, _: &Vec<Arc<Mutex<Filter>>>, _delta_time: Duration) {}
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
