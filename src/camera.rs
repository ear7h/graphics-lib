use glam::*;

#[derive(Debug)]
pub struct Camera {
    pub position : Vec3,
    pub forward : Vec3,
    pub up : Vec3,
    pub fov_y : f32,
    pub aspect : f32,
    pub near : f32,
    pub far : f32,
}

impl Camera {
    pub fn projection(&self) -> glam::Mat4 {
        glam::Mat4::perspective_rh_gl(
            self.fov_y,
            self.aspect,
            self.near,
            self.far,
        )
    }

    pub fn view(&self) -> glam::Mat4 {
        glam::Mat4::look_at_rh(
            self.position,
            self.forward,
            self.up,
        )
    }
}
