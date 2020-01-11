use std::any::TypeId;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;

use mopa::Any;

use crate::core::system_manager::Filter;

pub trait Component: mopa::Any + ComponentClone {}
mopafy!(Component);

// src: https://www.howtobuildsoftware.com/index.php/how-do/vk6/struct-clone-rust-traits-cloneable-how-to-clone-a-struct-storing-a-trait-object
pub trait ComponentClone {
    fn clone_box(&self) -> Box<dyn Component>;
}

impl<T> ComponentClone for T where T: 'static + Component + Clone {
    fn clone_box(&self) -> Box<dyn Component> {
        Box::new(self.clone())
    }
}

static mut LAST_ENTITY: AtomicU64 = AtomicU64::new(0);

pub type EntityID = u64;
pub type EntityRef = Arc<Mutex<Entity>>;

#[derive(Eq, PartialEq, Clone)]
pub struct Entity {
    pub id: EntityID,
    pub name: String,
    pub components: HashMap<TypeId, Box<dyn Component>>,
}

impl PartialEq for Box<dyn Component> {
    fn eq(&self, other: &Self) -> bool {
        self.get_type_id() == other.get_type_id()
    }
}

impl Eq for Box<dyn Component> {}

impl Clone for Box<dyn Component> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl Hash for Entity {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.name.hash(state);
    }
}

impl Entity {
    pub fn new() -> EntityRef {
        let val;
        unsafe {
            val = LAST_ENTITY.fetch_add(1, Relaxed);
        }
        Arc::new(Mutex::new(Entity {
            id: val,
            name: String::from("unnamed"),
            components: HashMap::new(),
        }))
    }

    pub fn get_component<T: Component>(&self) -> Result<&T, &str> {
        let contains = self.components.contains_key(&TypeId::of::<T>());
        return if contains {
            let tmp = self.components.get(&TypeId::of::<T>())
                .expect("Corrupt component");
            Ok(tmp.downcast_ref::<T>().expect("Corrupt component"))
        } else {
            Err("Component not contained.")
        };
    }

    pub fn add_component<T: Component>(&mut self, component: T) -> &mut Self {
        match self.components.insert(TypeId::of::<T>(), Box::new(component)) {
            Some(_) => (),
            None => ()
        }

        self
    }

    pub fn has_component<T: Component>(&self) -> bool {
        let contains = self.components.contains_key(&TypeId::of::<T>());
        return contains;
    }

    pub fn match_filter<>(&self, filter: &Filter) -> bool {
        filter.types.iter()
            .all(|t| { self.components.contains_key(t) })
    }
}

pub struct EntityManager {
    pub entities: HashMap<EntityID, EntityRef>,
    _last_id: EntityID,
}

impl EntityManager {
    pub fn new() -> EntityManager {
        EntityManager { entities: std::collections::HashMap::new(), _last_id: 0 }
    }

    pub fn add_entity(&mut self, e: EntityRef) {
        self.entities.insert(e.lock().unwrap().id, e.clone());
    }

    pub fn remove_entity(&mut self, e: EntityRef) {
        self.entities.remove(&e.lock().unwrap().id);
    }
}

