use glow::HasContext;
use quick_from::QuickFrom;
use std::rc::Rc;

pub struct LoadedProg {
    pub(crate) prog : glow::NativeProgram,
    pub(crate) active_uniforms : Rc<[(glow::ActiveUniform, glow::UniformLocation)]>,
}

impl LoadedProg {
    pub(crate) fn find_active_uniform(
        &self,
        name : &str
    ) -> Option<&(glow::ActiveUniform, glow::UniformLocation)> {
        self.active_uniforms.binary_search_by_key(
            &name,
            |v| &v.0.name
        ).ok().map(|idx| &self.active_uniforms[idx])

    }
}

pub trait Uniforms {
    fn set_uniforms(&self, setter : &mut UniformSetter<'_>);
}

impl<'a, T> Uniforms for &[(&str, T)]
where
    T : Into<UniformValue<'a>> + Clone
{
    fn set_uniforms(&self, setter : &mut UniformSetter<'_>) {
        for (name, val) in self.iter().cloned() {
            setter.set(name, val)
        }
    }
}

#[derive(QuickFrom, Clone)]
pub enum UniformValue<'a> {
    #[quick_from]
    Int(i32),

    #[quick_from]
    Uint(u32),

    #[quick_from]
    Intv(&'a [i32]),

    #[quick_from]
    Uintv(&'a [u32]),

    #[quick_from]
    Bool(bool),

    #[quick_from]
    Float(f32),

    #[quick_from]
    Vec2(glam::Vec2),
    #[quick_from]
    Vec3(glam::Vec3),
    #[quick_from]
    Vec4(glam::Vec4),

    #[quick_from]
    Vec2v(&'a [glam::Vec2]),
    #[quick_from]
    Vec3v(&'a [glam::Vec3]),
    #[quick_from]
    Vec4v(&'a [glam::Vec4]),

    #[quick_from]
    Mat2(glam::Mat2),
    #[quick_from]
    Mat3(glam::Mat3),
    #[quick_from]
    Mat4(glam::Mat4),

    #[quick_from]
    Mat2v(&'a [glam::Mat2]),
    #[quick_from]
    Mat3v(&'a [glam::Mat3]),
    #[quick_from]
    Mat4v(&'a [glam::Mat4]),
}

impl UniformValue<'_> {
    // TODO: handle samplers
    #[allow(dead_code)]
    pub(crate) fn gl_type(&self) -> u32 {
        use UniformValue::*;

        match self {
            Int(_) | Intv(_) => glow::INT,
            Uint(_) | Uintv(_)  => glow::INT,
            Bool(_) => glow::BOOL,
            Float(_) => glow::FLOAT,

            Vec2(_) | Vec2v(_) => glow::FLOAT_VEC2,
            Vec3(_) | Vec3v(_) => glow::FLOAT_VEC3,
            Vec4(_) | Vec4v(_) => glow::FLOAT_VEC4,

            Mat2(_) | Mat2v(_) => glow::FLOAT_MAT2,
            Mat3(_) | Mat3v(_) => glow::FLOAT_MAT3,
            Mat4(_) | Mat4v(_) => glow::FLOAT_MAT4,
        }
    }

    pub(crate) fn gl_size(&self) -> usize {
        use UniformValue::*;

        match self {
            Vec2v(v) => v.len(),
            Vec3v(v) => v.len(),
            Vec4v(v) => v.len(),
            Mat2v(v) => v.len(),
            Mat3v(v) => v.len(),
            Mat4v(v) => v.len(),
            _ => 1,
        }
    }

    pub(crate) fn set_uniform(
        self,
        gl : &glow::Context,
        loc : glow::UniformLocation
    ) {
        use UniformValue::*;

        unsafe {
        match self {
            Int(val) =>  {
                gl.uniform_1_i32_slice(
                    Some(&loc),
                    &[val],
                )
            },
            Uint(val) =>  {
                gl.uniform_1_u32_slice(
                    Some(&loc),
                    &[val],
                )
            },
            Intv(val) =>  {
                gl.uniform_1_i32_slice(
                    Some(&loc),
                    val,
                )
            },
            Uintv(val) =>  {
                gl.uniform_1_u32_slice(
                    Some(&loc),
                    val,
                )
            },
            Bool(val) =>  {
                gl.uniform_1_i32_slice(
                    Some(&loc),
                    &[if val { 1 } else { 0 }],
                )
            },
            Float(val) =>  {
                gl.uniform_1_f32_slice(
                    Some(&loc),
                    bytemuck::cast_slice(&[val])
                )
            },
            Vec2(val) =>  {
                gl.uniform_2_f32_slice(
                    Some(&loc),
                    bytemuck::cast_slice(&[val])
                )
            },
            Vec3(val) =>  {
                gl.uniform_3_f32_slice(
                    Some(&loc),
                    bytemuck::cast_slice(&[val])
                )
            },
            Vec4(val) =>  {
                gl.uniform_4_f32_slice(
                    Some(&loc),
                    bytemuck::cast_slice(&[val])
                )
            },
            Mat2(val) =>  {
                gl.uniform_matrix_2_f32_slice(
                    Some(&loc),
                    false,
                    bytemuck::cast_slice(&[val])
                )
            },
            Mat3(val) =>  {
                gl.uniform_matrix_3_f32_slice(
                    Some(&loc),
                    false,
                    bytemuck::cast_slice(&[val])
                )
            },
            Mat4(val) =>  {
                gl.uniform_matrix_4_f32_slice(
                    Some(&loc),
                    false,
                    bytemuck::cast_slice(&[val])
                )
            },

            Vec2v(val) =>  {
                gl.uniform_2_f32_slice(
                    Some(&loc),
                    bytemuck::cast_slice(val)
                )
            },
            Vec3v(val) =>  {
                gl.uniform_3_f32_slice(
                    Some(&loc),
                    bytemuck::cast_slice(val)
                )
            },
            Vec4v(val) =>  {
                gl.uniform_4_f32_slice(
                    Some(&loc),
                    bytemuck::cast_slice(val)
                )
            },

            Mat2v(val) =>  {
                gl.uniform_matrix_2_f32_slice(
                    Some(&loc),
                    false,
                    bytemuck::cast_slice(val)
                )
            },
            Mat3v(val) =>  {
                gl.uniform_matrix_3_f32_slice(
                    Some(&loc),
                    false,
                    bytemuck::cast_slice(val)
                )
            },
            Mat4v(val) =>  {
                gl.uniform_matrix_4_f32_slice(
                    Some(&loc),
                    false,
                    bytemuck::cast_slice(val)
                )
            },
        }
        }
    }
}

pub struct UniformSetter<'a> {
    pub(crate) gl : &'a glow::Context,
    pub(crate) prog : &'a LoadedProg,
}


impl<'a> UniformSetter<'a> {
    pub fn set<'b, V >(&self, name : &str, val : V)
    where
        V : Into<UniformValue<'b>>
    {
        let val : UniformValue = val.into();

        let meta_res = self.prog
            .find_active_uniform(name);

        // TODO: remove/make configurale
        let meta = match meta_res {
            Some(v) => v,
            None => {
                let msg = format!("no uniform {}", name);
                println!("{}", msg);
                return;
                // panic!("{}", msg)
            }
        };

        /* TODO: handle samplers
        if meta.0.utype != val.gl_type() {
            panic!(
                "destination type mismatch {}, expected {:x} got {:x}",
                name,
                meta.0.utype,
                val.gl_type(),
            );
        }
        */

        if (val.gl_size() as i32) > meta.0.size {
            panic!("size mismatch {}", name);
        }

        val.set_uniform(&self.gl, meta.1);

        error_check(&self.gl);
    }
}

fn error_check(gl : &glow::Context) {
    assert_eq!(
        unsafe { gl.get_error() },
        glow::NO_ERROR,
        "OpenGL error occurred!"
    );
}
