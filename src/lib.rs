use crate::core::EntityManager;

pub mod core;

pub struct NSE {
    pub entity_manager: EntityManager,
}

impl NSE {

    pub fn new() -> NSE {
        NSE {
            entity_manager: EntityManager::new()
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
