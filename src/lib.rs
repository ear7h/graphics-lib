pub mod camera;
use camera::*;

pub mod utils;
use utils::{
    create_display,
};

pub mod program;
use program::{
    LoadedProg,
    Uniforms,
    UniformValue,
    UniformSetter,
};

pub mod object;
use object::{
    LoadedObj,
    Obj,
};

pub mod scene_graph;

use std::cell::Cell;

use glam::{
    Vec3,
    Vec4,
    Mat4,
};

use glow::HasContext;

pub struct TextureDebug {
    prog : LoadedProg,
    plane : LoadedObj,
}

impl TextureDebug {
    pub fn new(ctx : &GraphicsContext) -> Self {
        println!("texture debug");
        let prog = ctx.load_program(
            include_str!("shaders/texture_dbg.vert"),
            include_str!("shaders/texture_dbg.frag"),
        ).unwrap();

        let plane = ctx.load_object(
            &Obj::plane()
        );

        Self{prog, plane}
    }

    pub fn render(
        &self,
        ctx : &GraphicsContext,
        // tex : glow::Texture,
        cache : &RenderCache,
        shadow_idx : usize,
    ) {
        let tex = cache.shadow_textures[shadow_idx];

        unsafe {

            ctx.gl.use_program(Some(self.prog.prog));
            ctx.gl.bind_vertex_array(Some(self.plane.vao));

            error_check(&ctx.gl);

            ctx.gl.active_texture(glow::TEXTURE0);
            ctx.gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            // bind a sampler object?

            let u = &[
                ("tex", UniformValue::Int(0)),
                ("near", UniformValue::Float(0.1)),
                ("far", UniformValue::Float(10.0)),
                // ("idx", idx as u32),
            ][..];

            u.set_uniforms(&mut UniformSetter{
                gl : &ctx.gl,
                prog : &self.prog,
            });


            error_check(&ctx.gl);

            ctx.gl.draw_elements(
                glow::TRIANGLES,
                self.plane.count as i32,
                glow::UNSIGNED_INT,
                0
            );

            ctx.gl.use_program(None);
            ctx.gl.bind_vertex_array(None);

            error_check(&ctx.gl);
        }
    }
}

pub struct ShadowMapper {
    prog : LoadedProg,
    fbo : glow::Framebuffer,
}

impl ShadowMapper {
    fn new_none_texture(ctx : &GraphicsContext) -> glow::Texture {
        unsafe {
            let tex = ctx.gl.create_texture().unwrap();

            ctx.gl.bind_texture(
                glow::TEXTURE_2D,
                Some(tex),
            );

            ctx.gl.tex_image_2d(
                glow::TEXTURE_2D, // target
                0, // level
                glow::DEPTH_COMPONENT24 as i32, // internalformat
                1, // width
                1, // heigh
                // MAX_SHADOWS as i32,
                0, // border
                glow::DEPTH_COMPONENT, // format
                glow::FLOAT, // type
                Some(bytemuck::cast_slice(&[f32::INFINITY])), // data
            );

            ctx.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32, // TODO: linear
            );

            ctx.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32, // TODO: linear
            );

            // no need to set border color since it defaults to 0.0

            ctx.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_BORDER as i32,
            );

            ctx.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_BORDER as i32,
            );

            ctx.gl.bind_texture(
                glow::TEXTURE_2D,
                None,
            );

            tex
        }
    }

    fn new_texture(ctx : &GraphicsContext) -> glow::Texture {
        unsafe {
            let tex = ctx.gl.create_texture().unwrap();

            ctx.gl.bind_texture(
                glow::TEXTURE_2D,
                Some(tex),
            );

            ctx.gl.tex_image_2d(
                glow::TEXTURE_2D, // target
                0, // level
                glow::DEPTH_COMPONENT24 as i32, // internalformat
                1024, // width
                1024, // heigh
                // MAX_SHADOWS as i32,
                0, // border
                glow::DEPTH_COMPONENT, // format
                glow::FLOAT, // type
                None, // data
            );

            ctx.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32, // TODO: linear
            );

            ctx.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32, // TODO: linear
            );

            ctx.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_BORDER as i32,
            );

            ctx.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_BORDER as i32,
            );

            ctx.gl.bind_texture(
                glow::TEXTURE_2D,
                None,
            );

            tex
        }
    }

    pub fn new(ctx : &GraphicsContext) -> Self {
        let prog = ctx.load_program(
            include_str!("shaders/shadow.vert"),
            include_str!("shaders/shadow.frag"),
        ).unwrap();

        let fbo = unsafe {
            ctx.gl.create_framebuffer().unwrap()
        };

        Self{ prog, fbo }

    }

    fn render_light(
        &self,
        ctx : &GraphicsContext,
        // idx : usize,
        // view : Mat4,
        tex : glow::Texture,
        lightspace : Mat4,
        scene : &scene_graph::Scene<'_, Light, Surface, LoadedObj>,
    ) {
        unsafe {

            ctx.gl.bind_framebuffer(
                glow::FRAMEBUFFER,
                Some(self.fbo),
            );

            ctx.gl.bind_texture(
                glow::TEXTURE_2D,
                Some(tex),
            );

            ctx.gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::DEPTH_ATTACHMENT,
                glow::TEXTURE_2D,
                Some(tex),
                0, // top level
                // 0, // idx as i32,
            );


            let fb_status = ctx.gl.check_framebuffer_status(
                glow::FRAMEBUFFER
            );

            assert_eq!(fb_status, glow::FRAMEBUFFER_COMPLETE);

            let mut old_vp = [0i32;4];
            ctx.gl.get_parameter_i32_slice(glow::VIEWPORT,  &mut old_vp);
            ctx.gl.viewport(0, 0, 1024, 1024);
            ctx.gl.draw_buffer(glow::NONE);
            ctx.gl.enable(glow::DEPTH_TEST);
            // I'm not sure where the scissor test is being enabled.
            // A quick look at egui_glow shows that it is disabled wherever it's
            // enabled. I should track it down with something like dbg!(check_scisor(ctx))
            // in various points of the update function
            ctx.gl.disable(glow::SCISSOR_TEST);
            ctx.gl.clear(glow::DEPTH_BUFFER_BIT);
            ctx.gl.use_program(Some(self.prog.prog));

            // iterate the objects
            scene.visit_surfaces(
                &mut |mat, s : &Surface, object : &LoadedObj| {
                    if !s.cast_shadow {
                        return
                    }

                    let u = &[
                        ("lightspace", UniformValue::Mat4(lightspace * mat )),
                    ][..];

                    u.set_uniforms(&mut UniformSetter{
                        gl : &ctx.gl,
                        prog : &self.prog,
                    });

                    error_check(&ctx.gl);

                    // TODO: the scene graph is optimized to group together
                    // calls with the same object, only call bind_vertex_array
                    // when he vao changes
                    ctx.gl.bind_vertex_array(Some(object.vao));

                    error_check(&ctx.gl);

                    ctx.gl.draw_elements(
                        glow::TRIANGLES,
                        object.count as i32,
                        glow::UNSIGNED_INT,
                        0
                    );
                }
            );

            ctx.gl.use_program(None);
            ctx.gl.bind_vertex_array(None);
            ctx.gl.disable(glow::DEPTH_TEST);
            ctx.gl.enable(glow::SCISSOR_TEST);

            error_check(&ctx.gl);

            let [x, y, w, h] = old_vp;
            ctx.gl.viewport(x, y, w, h);
            ctx.gl.bind_framebuffer(
                glow::DRAW_FRAMEBUFFER,
                None,
            );

            error_check(&ctx.gl);
        }
    }
}

type GlutinContext = glutin::ContextWrapper<
    glutin::PossiblyCurrent,
    glutin::window::Window
>;

pub struct RenderCache {
    shadow_mapper : ShadowMapper,
    shadow_textures_idx : usize,
    shadow_textures : Vec<glow::Texture>,
    shadow_none_textures_idx : usize,
    shadow_none_textures : Vec<glow::Texture>,
}

impl RenderCache {
    pub fn new(ctx : &GraphicsContext) -> Self {
        Self{
            shadow_mapper : ShadowMapper::new(ctx),
            shadow_textures_idx : 0,
            shadow_textures : Vec::new(),
            shadow_none_textures_idx : 0,
            shadow_none_textures : Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.shadow_textures_idx = 0;
        self.shadow_none_textures_idx = 0;
    }

    fn alloc_shadow_texture(
        &mut self,
        ctx : &GraphicsContext
    ) -> glow::Texture {

        assert!(self.shadow_textures.len() >= self.shadow_textures_idx);
        if self.shadow_textures.len() > self.shadow_textures_idx {
            self.shadow_textures[self.shadow_textures_idx]
        } else {
            self.shadow_textures.push(ShadowMapper::new_texture(&ctx));
            self.shadow_textures[self.shadow_textures_idx]
        }
    }

    fn alloc_shadow_none_texture(&mut self, ctx : &GraphicsContext) -> glow::Texture {
        assert!(self.shadow_none_textures.len() >= self.shadow_none_textures_idx);
        if self.shadow_none_textures.len() > self.shadow_textures_idx {
            self.shadow_none_textures[self.shadow_none_textures_idx]
        } else {
            self.shadow_none_textures.push(ShadowMapper::new_none_texture(&ctx));
            self.shadow_none_textures[self.shadow_none_textures_idx]
        }
    }
}

pub struct GraphicsContext {
    gl : glow::Context,
    gl_window : GlutinContext,
    egui : egui_glow::EguiGlow,
}

impl GraphicsContext {
    pub fn render_egui<T>(
        &mut self,
        mut f : impl FnMut(&egui::CtxRef) -> T
    ) -> T {
        self.egui.begin_frame(self.gl_window.window());

        let ret = (f)(self.egui.ctx());

        let (needs_repaint, shapes) = self.egui.end_frame(
            self.gl_window.window()
        );

        if needs_repaint {
            self.gl_window.window().request_redraw();
        }

        self.egui.paint(&self.gl_window, &self.gl, shapes);

        ret
    }

    pub fn set_title(&self, s : &str) {
        self.gl_window.window().set_title(s);
    }

    // TODO: reuturn a Size with an aspect method
    pub fn aspect(&self) -> f32 {
        let scale = self.gl_window.window().scale_factor();
        let size : glutin::dpi::LogicalSize<f32> = self.gl_window.window().inner_size().to_logical(scale);
        (size.width as f32) / (size.height as f32)
    }

    pub fn logical_size(&self) -> (u32, u32) {
        let scale = self.gl_window.window().scale_factor();
        let size = self.gl_window.window().inner_size().to_logical(scale);

        (size.width, size.height)
    }

    pub fn swap_buffers(&self) {
        self.gl_window.swap_buffers().unwrap();
    }

    pub fn load_object(&self, obj : &Obj) -> LoadedObj {
        let vao;
        let index_vbo;
        let vertex_vbo;
        let normal_vbo;

        unsafe {
            vao = self.gl.create_vertex_array().unwrap();
            self.gl.bind_vertex_array(Some(vao));
        }

        error_check(&self.gl);

        // load vertices
        unsafe {
            vertex_vbo = self.gl.create_buffer().unwrap();
            self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(vertex_vbo));
            self.gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&obj.vertices),
                glow::STATIC_DRAW,
            );
            self.gl.enable_vertex_attrib_array(0);

            self.gl.vertex_attrib_pointer_f32(
                0,
                3,
                glow::FLOAT,
                false,
                0,
                0,
            );

        }

        // load normals
        unsafe {
            normal_vbo = self.gl.create_buffer().unwrap();
            self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(normal_vbo));
            self.gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&obj.normals),
                glow::STATIC_DRAW,
            );
            self.gl.enable_vertex_attrib_array(1);
            self.gl.vertex_attrib_pointer_f32(
                1,
                3,
                glow::FLOAT,
                false,
                0,
                0,
            );
        }

        // load indices
        unsafe {
            index_vbo = self.gl.create_buffer().unwrap();
            self.gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(index_vbo));
            self.gl.buffer_data_u8_slice(
                glow::ELEMENT_ARRAY_BUFFER,
                bytemuck::cast_slice(&obj.indices),
                glow::STATIC_DRAW,
            );
        }

        unsafe {
            self.gl.bind_vertex_array(None);
        }

        error_check(&self.gl);

        LoadedObj{
            vao, index_vbo, vertex_vbo, normal_vbo,
            count : obj.indices.len(),
        }
    }

    pub fn unload_object(&self, _obj : LoadedObj) {
        todo!()
    }

    pub fn load_program(
        &self,
        vert_src : &str,
        frag_src : &str,
    ) -> Result<LoadedProg, String> {
        println!("compiling vert");
        let vert;
        unsafe {
            vert = self.gl.create_shader(glow::VERTEX_SHADER).unwrap();
            self.gl.shader_source(vert, &vert_src);
            self.gl.compile_shader(vert);
            if !self.gl.get_shader_compile_status(vert) {
                let s = self.gl.get_shader_info_log(vert);
                self.gl.delete_shader(vert);
                return Err(s)
            }
        }

        println!("compiling frag");
        let frag;
        unsafe {
            frag = self.gl.create_shader(glow::FRAGMENT_SHADER).unwrap();
            self.gl.shader_source(frag, &frag_src);
            self.gl.compile_shader(frag);
            if !self.gl.get_shader_compile_status(frag) {
                let s = self.gl.get_shader_info_log(frag);
                self.gl.delete_shader(frag);
                return Err(s)
            }
        }

        println!("compiling linking");
        let prog;
        unsafe {
            prog = self.gl.create_program().unwrap();
            self.gl.attach_shader(prog, vert);
            self.gl.attach_shader(prog, frag);
            self.gl.link_program(prog);

            self.gl.detach_shader(prog, vert);
            self.gl.detach_shader(prog, frag);
            self.gl.delete_shader(vert);
            self.gl.delete_shader(frag);
            if !self.gl.get_program_link_status(prog) {
                self.gl.delete_program(prog);
                return Err(self.gl.get_program_info_log(prog));
            }
        }

        let n = unsafe {
            self.gl.get_active_uniforms(prog)
        };

        let mut active_uniforms = Vec::with_capacity(n as usize);

        for i in 0..n {
            let u = unsafe {
                self.gl.get_active_uniform(
                    prog,
                    i
                ).unwrap()
            };

            println!("uniform name: {}", u.name);

            let loc = unsafe {
                self.gl.get_uniform_location(
                    prog,
                    &u.name,
                ).unwrap()
            };

            active_uniforms.push((u, loc));
        }

        active_uniforms.sort_unstable_by(|left, right| {
            left.0.name.cmp(&right.0.name)
        });

        Ok(LoadedProg{
            prog,
            active_uniforms: active_uniforms.into_boxed_slice().into(),
        })
    }

    pub fn unload_program(
        &self,
        _prog : LoadedProg,
    ) {
        todo!();
    }

    fn set_texture_parameters(&self) {
        unsafe {
            self.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );

            self.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );

            self.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );

            self.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
        }
    }

    pub fn texture_2d_image(
        &self,
        img : &image::RgbImage,
    ) -> glow::Texture {

        unsafe {
            let tex = self.gl.create_texture().unwrap();
            self.gl.bind_texture(glow::TEXTURE_2D, Some(tex));

            self.gl.tex_image_2d(
                glow::TEXTURE_2D, // target
                0, // level
                glow::RGB as i32, // internalformat
                img.width() as i32,
                img.height() as i32,
                0, // border
                glow::RGB, // format
                glow::UNSIGNED_BYTE, // type
                Some(img.as_raw()), // data
            );

            self.set_texture_parameters();

            self.gl.bind_texture(glow::TEXTURE_2D, None);

            tex
        }
    }

    pub fn clear(&self, r : f32, g : f32, b : f32, a : f32) {
        unsafe {
            self.gl.clear_color(r, g, b, a);
            self.gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
        }
    }

    /// major, minor
    pub fn gl_version(&self) -> (i32, i32) {
        unsafe {
            let major = self.gl.get_parameter_i32(
                glow::MAJOR_VERSION,
            );

            let minor = self.gl.get_parameter_i32(
                glow::MAJOR_VERSION,
            );

            (major, minor)
        }
    }

    pub fn render_scene<U : Uniforms>(
        &self,
        cache : &mut RenderCache,
        camera : &Camera,
        scene: &scene_graph::Scene<'_, Light, Surface, LoadedObj>,
        prog : &LoadedProg,
        extra_uniforms : U,
    ) {

        cache.clear();

        #[derive(Default)]
        struct LightUniforms {
            len : i32,

            positions : Vec<Vec4>,
            colors : Vec<Vec4>,
            lightspaces : Vec<Mat4>,
            shadows : Vec<i32>,
        }

        // TODO: move this to the cache
        let mut shadows = Vec::new();
        let mut lights = LightUniforms::default();

        scene.visit_lights(
            &mut |mat, l|  {
                lights.len += 1;

                lights.positions.push(
                    mat * l.view() * Vec4::W,
                );

                lights.colors.push(
                    l.color(),
                );

                shadows.push((l.view() * mat.inverse(), l.shadow()));
            }
        );

        // TODO: move this to the cache
        let mut texture_loads = Vec::new();
        let mut texture_unit = 0i32..;

        let none_texture_unit = texture_unit.next().unwrap();

        texture_loads.push((
                none_texture_unit,
                cache.alloc_shadow_none_texture(self),
        ));

        for (mat, shadow) in shadows.drain(..) {
            match shadow {
                Shadow::None(proj) => {
                    let lightspace = proj * mat;

                    lights.lightspaces.push(lightspace);
                    lights.shadows.push(none_texture_unit);
                },
                Shadow::Plane(proj) => {

                    let tex = cache.alloc_shadow_texture(self);
                    let lightspace = proj * mat;

                    // TODO?: the shadow mapper uses a texture unit which is known
                    // to us (currently unit 0), so we could start loading
                    // texture units right away
                    cache.shadow_mapper.render_light(
                        self,
                        // camera.view(),
                        tex,
                        lightspace,
                        scene,
                    );

                    let unit = texture_unit.next().unwrap();

                    texture_loads.push((
                            unit,
                            tex,
                    ));

                    lights.lightspaces.push(lightspace);
                    lights.shadows.push(unit);
                },
            }
        }


        struct SceneUniforms<U> {
            // position and color
            lights : LightUniforms,
            // if true then the above uniforms are not set
            projection : glam::Mat4,
            inited : Cell<bool>,

            model : glam::Mat4,
            view : glam::Mat4,
            phong : PhongMaterial,
            extra : U,
        }

        impl<U> Uniforms for SceneUniforms<U>
        where
            U : Uniforms
        {
            fn set_uniforms(&self, s : &mut UniformSetter<'_>) {
                macro_rules! set {
                    ($($id:tt)*) => {
                        s.set(stringify!($($id)*), self.$($id)*);
                    }
                }

                macro_rules! set_slice {
                    ($($id:tt)*) => {
                        s.set(concat!(stringify!($($id)*), "[0]"), self.$($id)*.as_slice());
                    }
                }

                if !self.inited.get() {
                    self.extra.set_uniforms(s);


                    set_slice!(lights.lightspaces);
                    set_slice!(lights.shadows);
                    set_slice!(lights.positions);
                    set_slice!(lights.colors);
                    set!(lights.len);

                    set!(projection);
                    self.inited.set(true);
                }

                set!(model);
                set!(view);
                set!(projection);
                set!(phong.ambient);
                set!(phong.diffuse);
                set!(phong.specular);
                set!(phong.emission);
                set!(phong.shininess);
            }
        }

        let mut uniforms = SceneUniforms {
            lights,
            projection : camera.projection(),
            inited : Default::default(),

            view : camera.view(),
            model : glam::Mat4::IDENTITY,
            phong : Default::default(),
            extra : extra_uniforms,
        };


        unsafe {
            self.gl.use_program(Some(prog.prog));
            self.gl.enable(glow::DEPTH_TEST);
            let (w, h) = self.logical_size();
            self.gl.viewport(0, 0, w.try_into().unwrap(), h.try_into().unwrap());

            for (unit, tex) in texture_loads {
                self.gl.active_texture(glow::TEXTURE0 + (unit as u32));
                self.gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                error_check(&self.gl);
            }
        }

        error_check(&self.gl);

        let mut prev_vao = None;

        scene.visit_surfaces(
            &mut |mat, surface : &Surface, object : &LoadedObj| {
                uniforms.model = mat;
                uniforms.phong = surface.material;

                uniforms.set_uniforms(&mut UniformSetter{
                    gl : &self.gl,
                    prog : prog,
                });

                error_check(&self.gl);

                unsafe {
                    // TODO: benchmark this optimization
                    //
                    // the scene graph is optimized to group together calls
                    // with the same object
                    //
                    // note: to benchmark this, a new naive scene graph needs
                    // to be implemented, not just removing the conditional,
                    // because the underlying OpenGL implementation might
                    // have this check as well
                    let vao = Some(object.vao);
                    if vao != prev_vao {
                        self.gl.bind_vertex_array(vao);
                        prev_vao = vao;
                        error_check(&self.gl);
                    }

                    self.gl.draw_elements(
                        glow::TRIANGLES,
                        object.count as i32,
                        glow::UNSIGNED_INT,
                        0
                    );
                }
            }
        );

        unsafe {
            for unit in 0..texture_unit.next().unwrap() {
                self.gl.active_texture(glow::TEXTURE0 + (unit as u32));
                self.gl.bind_texture(glow::TEXTURE_2D, None);
            }

            self.gl.use_program(None);
            self.gl.bind_vertex_array(None);
            self.gl.disable(glow::DEPTH_TEST);
            assert_eq!(
                self.gl.get_error(),
                glow::NO_ERROR,
                "OpenGL error occurred!"
            );
        }
    }
}


fn error_check(gl : &glow::Context) {
    assert_eq!(
        unsafe { gl.get_error() },
        glow::NO_ERROR,
        "OpenGL error occurred!"
    );
}


#[derive(Clone, Copy, Default)]
pub struct PhongMaterial {
    pub ambient : glam::Vec4,
    pub diffuse : glam::Vec4,
    pub specular : glam::Vec4,
    pub emission : glam::Vec4,
    pub shininess : f32,
}

impl PhongMaterial {
    pub fn turquoise() -> PhongMaterial {
        PhongMaterial{
            ambient : glam::Vec4::new(0.1, 0.2, 0.17, 1.0),
            diffuse : glam::Vec4::new(0.2, 0.375, 0.35, 1.0),
            specular : glam::Vec4::new(0.3, 0.3, 0.3, 1.0),
            emission : glam::Vec4::W,
            shininess : 100.0,
        }
    }
}


#[derive(Clone, Copy)]
pub enum Light {
    Point{
        shadow : bool,
        color : glam::Vec4,
        // TODO: shadows
    },
    Directional {
        shadow : bool,
        color : glam::Vec4,
        left : f32,
        right : f32,
        bottom : f32,
        top : f32,
        near : f32,
        far : f32,
    },
    Spot{
        shadow : bool,
        color : glam::Vec4,
        forward : glam::Vec3,
        angle : f32,
        near : f32,
        far : f32,
    },
}

pub enum Shadow{
    None(Mat4),
    Plane(Mat4),
    // TODO: use for point lights
    // Cube([Mat4;6]),
}

impl Light {
    fn view(&self) -> glam::Mat4 {
        use Light::*;

        match self {
            Spot{forward, ..} => {
                Mat4::look_at_rh(
                    Vec3::ZERO,
                    *forward,
                    forward.any_orthogonal_vector(),
                )
            },
            _ => Mat4::IDENTITY,
        }
    }

    fn color(&self) -> glam::Vec4 {
        use Light::*;

        match self {
            | Point{color, ..}
            | Directional{color, ..}
            | Spot{color, ..} => *color,
        }
    }

    fn shadow(&self) -> Shadow {
        use Light::*;
        match self {
            Point{shadow, ..} => {
                // Shadow::None,
                if *shadow {
                    todo!("set up a (fake) cube shadow map for point lights")
                } else {
                    Shadow::None(Mat4::ZERO)
                }
            },
            Directional{
                shadow,
                left, right,
                bottom, top,
                near, far,
                ..
            } => {
                let mat = Mat4::orthographic_rh(
                    *left, *right,
                    *bottom, *top,
                    *near, *far
                );

                if *shadow {
                    Shadow::Plane(mat)
                } else {
                    Shadow::None(mat)
                }
            },
            Spot{shadow, angle, near, far, ..} => {
                let mat = Mat4::perspective_rh(
                    *angle, 1.0,
                    *near, *far
                );

                if *shadow {
                    Shadow::Plane(mat)
                } else {
                    Shadow::None(mat)
                }
            },
        }
    }
}

#[derive(Clone, Copy)]
pub struct Surface {
    pub cast_shadow : bool,
    pub material : PhongMaterial,
}

pub trait App {
    // load shaders, textures, etc.
    fn init(ctx : &mut GraphicsContext) -> Self;

    // do rendering
    fn update(
        &mut self,
        event : glutin::event::Event<'_, ()>,
        ctx : &mut GraphicsContext,
        control_flow : &mut glutin::event_loop::ControlFlow,
    );
}


pub fn is_keyboard_event<'a, T>(
    event: &'a glutin::event::Event<'_, T>
) -> Option<&'a glutin::event::KeyboardInput> {
    use glutin::event::*;

    if let Event::WindowEvent{
        event : WindowEvent::KeyboardInput{ input, ..},
        ..
    } = event {
        Some(input)
    } else {
        None
    }
}

pub fn is_redraw_event<T>(event : &glutin::event::Event<'_, T>) -> bool {
    // Platform-dependent event handlers to workaround a winit bug
    // See: https://github.com/rust-windowing/winit/issues/987
    // See: https://github.com/rust-windowing/winit/issues/1619
    match event {
        glutin::event::Event::RedrawEventsCleared if cfg!(windows) => true,
        glutin::event::Event::RedrawRequested(_) if !cfg!(windows) => true,
        _ => false,
    }
}

pub fn run<A : App + 'static>() {
    println!("Hello, world!");


    let event_loop = glutin::event_loop::EventLoop::with_user_event();
    let (gl_window, gl) = create_display(&event_loop);

    let egui = egui_glow::EguiGlow::new(&gl_window, &gl);

    let mut render_ctx = GraphicsContext {
        gl, gl_window, egui
    };

    let mut a = A::init(&mut render_ctx);


    event_loop.run(move |event, _, control_flow| {
        if let glutin::event::Event::WindowEvent { ref event, .. } = event {
            if render_ctx.egui.is_quit_event(&event) {
                *control_flow = glutin::event_loop::ControlFlow::Exit;
            }

            if let glutin::event::WindowEvent::Resized(physical_size) = event {
                render_ctx.gl_window.resize(*physical_size);
            }

            render_ctx.egui.on_event(&event);

            render_ctx.gl_window.window().request_redraw(); // TODO: ask egui if the events warrants a repaint instead
        }

        if let glutin::event::Event::LoopDestroyed = event {
            render_ctx.egui.destroy(&render_ctx.gl);
            // TODO tear down GL
        }

        a.update(event, &mut render_ctx, control_flow);
    });
}

