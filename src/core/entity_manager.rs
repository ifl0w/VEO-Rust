use std::collections::HashSet;

type EntityID = u64;

#[derive(Eq, PartialEq, Hash, Clone)]
pub struct Entity {
    id: EntityID,
    pub name: String
}

impl Entity {
    pub fn new() -> Entity {
        Entity {
            id: 0,
            name: String::from("unnamed")
        }
    }
}

pub struct EntityManager {
    entities: HashSet<Entity>,
    _last_id: u64
}

impl EntityManager {

    pub fn new() -> EntityManager {
        EntityManager { entities: std::collections::HashSet::new(), _last_id: 0 }
    }

    pub fn add_entity(&mut self, e: &Entity) {
        println!("adding entity");
        self.entities.insert(e.clone());
    }

    pub fn remove_entity(&mut self, e: &Entity) {
        println!("removing entity");
        self.entities.remove(e);
    }

    pub fn list_entities(&self) {
        for e in self.entities.iter() {
            println!("{}", e.name);
        }
    }

}

