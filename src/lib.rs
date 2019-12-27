use crate::core::EntityManager;
use crate::core::SystemManager;

pub mod core;

pub struct NSE {
    pub entity_manager: EntityManager,
    pub system_manager: SystemManager,
}

impl NSE {

    pub fn new() -> NSE {
        NSE {
            entity_manager: EntityManager::new(),
            system_manager: SystemManager::new()
        }
    }

    pub fn init() {
        println!("Initializing...")
    }

    pub fn run() {
        println!("Running...")
    }
}



#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
