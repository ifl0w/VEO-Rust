use std::any::TypeId;
use std::fmt::Debug;

use crossbeam::crossbeam_channel::unbounded;

pub trait Payload: Debug + PayloadClone + mopa::Any {}
mopafy!(Payload);

// src: https://www.howtobuildsoftware.com/index.php/how-do/vk6/struct-clone-rust-traits-cloneable-how-to-clone-a-struct-storing-a-trait-object
pub trait PayloadClone {
    fn clone_box(&self) -> Box<dyn Payload>;
}

impl<T> PayloadClone for T where T: 'static + Payload + Clone {
    fn clone_box(&self) -> Box<dyn Payload> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Message {
    pub code: TypeId,
    pub data: Box<dyn Payload>,
}

impl Clone for Box<dyn Payload> {
    fn clone(&self) -> Box<dyn Payload> {
        self.clone_box()
    }
}

impl Message {
    pub fn new<T: 'static + Payload>(payload: Box<T>) -> Self {
        Message {
            code: TypeId::of::<T>(),
            data: payload,
        }
    }

    pub fn is_type<T: 'static + Payload>(&self) -> bool {
        TypeId::of::<T>() == self.code
    }

    pub fn get_payload<T: 'static + Payload>(&self) -> Result<&T, &str> {
        if TypeId::of::<T>() == self.code {
            let tmp = self.data.as_ref().downcast_ref::<T>();
            let result = tmp.expect("Corrupt Message");
            Ok(result)
        } else {
            Err("Incorrect Type")
        }
    }
}

pub struct MessageManager {
    // maps message type to listener objects
    pub sender: crossbeam::crossbeam_channel::Sender<Message>,
    pub receiver: crossbeam::crossbeam_channel::Receiver<Message>,
}

impl MessageManager {
    pub fn new() -> Self {
        let (send, receive) = unbounded();

        MessageManager {
            sender: send,
            receiver: receive,
        }
    }
}


#[derive(Debug, Clone)]
pub struct Empty {}

impl Payload for Empty {}

#[derive(Debug, Clone)]
pub struct Exit {}

impl Payload for Exit {}

#[derive(Debug, Clone)]
pub struct Text {
    pub text: String
}

impl Payload for Text {}
