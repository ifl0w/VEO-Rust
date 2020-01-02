use std::collections::HashSet;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;

type EntityID = u64;


static mut LAST_ENTITY: AtomicU64 = AtomicU64::new(0);

#[derive(Eq, PartialEq, Hash, Clone)]
pub struct Entity {
    pub id: EntityID,
    pub name: String,
}

impl Entity {
    pub fn new() -> Entity {
        let val;
        unsafe {
            val = LAST_ENTITY.fetch_add(1, Relaxed);
        }
        Entity {
            id: val,
            name: String::from("unnamed"),
        }
    }
}

pub struct EntityManager {
    pub entities: HashSet<Box<Entity>>,
    _last_id: u64,
}

impl EntityManager {
    pub fn new() -> EntityManager {
        EntityManager { entities: std::collections::HashSet::new(), _last_id: 0 }
    }

    pub fn add_entity(&mut self, e: &Box<Entity>) {
        self.entities.insert(e.clone());
    }

    pub fn remove_entity(&mut self, e: &Box<Entity>) {
        self.entities.remove(e);
    }
}

