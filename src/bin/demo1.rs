use game_engine::*;

use camera::*;

use program::{
    LoadedProg,
};

use object::{
    LoadedObj,
    Obj,
};

use scene_graph::{
    NodeHandle,
    Node,
};

use std::time::{Instant, Duration};
use std::io;
use std::fs::File;
use std::f32::consts::PI;

use winit_input_helper::WinitInputHelper;

use rand::distributions::Distribution;
use rand::SeedableRng;

use glam::{
    Vec3,
    Vec4,
    Mat4,
};


#[derive(Clone, Copy, Debug)]
pub struct SphereCoord {
    r : f32,
    theta : f32,
    phi : f32,
}

impl SphereCoord {
    fn matrix(&self) -> Mat4 {
        Mat4::from_axis_angle(Vec3::Y, self.phi) *
        Mat4::from_axis_angle(Vec3::X, -self.theta)
    }

    fn position(&self) -> Vec3 {
        self.matrix().transform_point3(Vec3::Z * self.r)
    }

    fn up(&self) -> Vec3 {
        self.matrix().transform_vector3(Vec3::Y)
    }
}

type SceneGraph = scene_graph::SceneGraph<Light, Surface, LoadedObj>;

struct Scene {
    graph : SceneGraph,
    floor : NodeHandle,
    sun : NodeHandle,
    buildings : [NodeHandle; 1],
}

impl Scene {
    fn new(ctx : &mut GraphicsContext) -> Self {

        let mut g = SceneGraph::default();

        #[cfg(feature = "static")]
        let f = include_str!("../models/sphere.obj").as_bytes();

        #[cfg(not(feature = "static"))]
        let f = File::open("models/sphere.obj").unwrap();

        let obj = Obj::parse(io::BufReader::new(f)).unwrap();
        let sphere = g.add_object(
            ctx.load_object(&obj)
        );

        let sun_sphere = g.add_surface(
            Surface{
                cast_shadow : false,
                material : PhongMaterial::turquoise(),
            },
            sphere,
        );

        const DIM : f32 = 100.0;

        let sun_light1 = g.add_light(
            Light::Directional{
                shadow : true,
                color : Vec4::new(1.0, 0.8, 0.0, 1.0),
                left : -DIM,
                right : DIM,
                bottom : -DIM,
                top : DIM,
                near : 10.0,
                far : 100.0,
            }
        );

        let sun_light2 = g.add_light(
            Light::Directional{
                shadow : true,
                color : Vec4::new(1.0, 1.0, 1.0, 1.0),
                left : -DIM,
                right : DIM,
                bottom : -DIM,
                top : DIM,
                near : 10.0,
                far : 100.0,
            }
        );

        let sun = g.add_branch(&[
            (
                Mat4::from_translation(
                    Vec3::new(0.0, 0.0, 35.0)
                ),
                sun_sphere,
            ),
            (
                Mat4::from_translation(
                    Vec3::new(0.0, 0.0, 30.0)
                ),
                sun_light1,
            ),
            (
                Mat4::from_translation(
                    Vec3::new(0.0, 0.0, 30.0)
                ),
                sun_light2,
            ),
        ]);

        let cube = g.add_object(
            ctx.load_object(&Obj::cube())
        );

        let gray = PhongMaterial {
            ambient : Vec4::new(0.1, 0.1, 0.1, 1.0),
            diffuse : Vec4::new(0.3, 0.3, 0.3, 1.0),
            specular : Vec4::new(0.2, 0.2, 0.2, 1.0),
            emission : glam::Vec4::W,
            shininess : 50.0,
        };

        let floor = g.add_surface(
            Surface{
                cast_shadow : true,
                material : gray,
            },
            cube,
        );

        let floor = g.add_branch(&[
            (
                Mat4::from_scale_rotation_translation(
                    Vec3::new(100.0, 1.0, 100.0),
                    glam::Quat::IDENTITY,
                    Vec3::new(0.0, -0.5, 0.0)
                ),
                floor,
            ),
        ]);

        let buildings = [
            g.add_surface(
                Surface{
                    cast_shadow : true,
                    material : gray,
                },
                cube,
            ),
        ];

        Self {
            graph : g,
            // cache : Default::default(),
            sun,
            floor,
            buildings,
        }
    }
}


struct Demo1 {
    values : [f32; 3],
    bools : [bool; 3],

    enable_lighting : bool,
    enable_gui : bool,

    prog : LoadedProg,

    camera : Camera,
    camera_pos : SphereCoord,

    input : WinitInputHelper,

    render_cache : RenderCache,

    shadow_map_debug: TextureDebug,

    scene : Scene,

    time : Instant,
}

impl App for Demo1 {
    fn init(ctx : &mut GraphicsContext) -> Self {
        // try frame buffer

        ctx.set_title("demo1 - city");

        let (major, minor) = ctx.gl_version();
        println!("OpenGL version: {}.{}", major, minor);

        #[cfg(feature = "static")]
        let prog_res = ctx.load_program(
            include_str!("../vert.vert"),
            include_str!("../frag.frag"),
        );

        #[cfg(not(feature = "static"))]
        let prog_res = ctx.load_program(
            &std::fs::read_to_string("vert.vert").unwrap(),
            &std::fs::read_to_string("frag.frag").unwrap(),
        );

        let prog = match prog_res {
            Ok(v) => v,
            Err(s) => {
                println!("SHADER ERROR: ");
                println!("{}", s);
                std::process::exit(1);
            }
        };

        let camera = Camera {
            position : glam::Vec3::new(0.0, 0.0, 5.0),
            forward : glam::Vec3::new(0.0, 0.0, 0.0),
            up : glam::Vec3::new(0.0, 1.0, 0.0),
            fov_y : std::f32::consts::PI / 2.0,
            aspect : 1.0,
            near : 0.01,
            far : 100.0,
        };

        let shadow_map_debug = TextureDebug::new(ctx);

        Self{
            values: [0.0;3],
            bools: [false;3],
            enable_lighting : true,
            enable_gui : true,
            prog,
            camera,
            camera_pos : SphereCoord {
                r : 35.0,
                theta: 0.5,
                phi : 0.5,
            },
            input : WinitInputHelper::new(),
            render_cache : RenderCache::new(ctx),
            shadow_map_debug,
            scene : Scene::new(ctx),
            time : Instant::now(),
        }
    }

    fn update(
        &mut self,
        event : glutin::event::Event<'_, ()>,
        ctx : &mut GraphicsContext,
        control_flow : &mut glutin::event_loop::ControlFlow,
    ) {
        let mut quit = false;

        if self.input.update(&event) {
            use glutin::event::VirtualKeyCode;

            let input = &self.input;

            if input.key_pressed(VirtualKeyCode::H) {
                self.enable_gui = !self.enable_gui;
            }

            if input.key_pressed(VirtualKeyCode::Q) {
                quit = true;
            }

            if input.key_pressed(VirtualKeyCode::L) {
                self.enable_lighting = !self.enable_lighting;
            }

            const DELTA : f32 = 0.5;

            if input.key_pressed(VirtualKeyCode::Up) {
                self.camera_pos.theta += DELTA;
            }

            if input.key_pressed(VirtualKeyCode::Down) {
                self.camera_pos.theta -= DELTA;
            }

            if input.key_pressed(VirtualKeyCode::Right) {
                self.camera_pos.phi += DELTA;
            }

            if input.key_pressed(VirtualKeyCode::Left) {
                self.camera_pos.phi -= DELTA;
            }
        }

        let now = Instant::now();
        // TODO: laptop can't handle 60fps + user input
        let next_draw = self.time + (Duration::from_secs(1) / 60);

        if is_redraw_event(&event) || (self.bools[0] && now >= next_draw) {
            ctx.clear(0.1, 0.1, 0.1, 1.0);

            let delta = now - self.time;
            self.time = now;

            const N : usize = 1020;

            let mut roots = Vec::with_capacity(N + 2);

            // animate sun
            if self.bools[0] {
                self.values[0] += delta.as_secs_f32() / 4.0;
                self.values[0] = self.values[0].fract();
            }

            roots.extend_from_slice(&[
                (
                    Mat4::IDENTITY,
                    Node::Handle(self.scene.floor),
                ),
                (
                    Mat4::from_scale_rotation_translation(
                        Vec3::ONE,
                        glam::Quat::from_rotation_z(
                            2.0 * PI * self.values[1]
                        ) * glam::Quat::from_rotation_x(
                            2.0 * PI * self.values[0]
                        ),
                        Vec3::new(0.0, 0.0, 0.0),
                    ),
                    Node::Handle(self.scene.sun),
                ),
            ]);

            const CITY_DIM : f32 = 20.0;

            let mut rng = rand::rngs::SmallRng::seed_from_u64(0xdeadbeef);
            let dist = statrs::distribution::Normal::new(
                0.0,
                (CITY_DIM / 3.0) as f64
            ).unwrap();

            for _ in 0..N {
                let x = dist.sample(&mut rng) as f32;
                let x = x
                    .max(-CITY_DIM)
                    .min(CITY_DIM);

                let y = dist.sample(&mut rng) as f32;
                let y = y
                    .max(-CITY_DIM)
                    .min(CITY_DIM);

                let height = bellish(x.abs() + y.abs()) * 20.0 + 0.5;

                roots.push(
                    (
                        Mat4::from_scale_rotation_translation(
                            Vec3::new(1.0, height, 1.0),
                            glam::Quat::IDENTITY,
                            Vec3::new(x, height / 2.0, y),
                        ),
                        Node::Handle(self.scene.buildings[0]),
                    )
                );
            }


            self.camera.aspect = ctx.aspect();
            self.camera.position = self.camera_pos.position();
            self.camera.up = self.camera_pos.up();

            ctx.render_scene(
                &mut self.render_cache,
                &self.camera,
                &scene_graph::Scene::new(
                    &self.scene.graph,
                    &roots,
                    &mut Default::default(),
                ),
                &self.prog,
                &[
                    ("enable_lighting", self.enable_lighting),
                ][..],
            );

            if self.bools[1] {
                self.shadow_map_debug.render(
                    ctx,
                    &self.render_cache,
                    0,
                );
            }

            ctx.render_egui(|ctx| {
                if !self.enable_gui {
                    return
                }

                egui::SidePanel::left("left_panel")
                .resizable(true)
                .show(ctx, |ui| {
                    ui.heading("City Demo");
                    quit = ui.button("Quit").clicked();

                    let pos = &mut self.camera_pos;

                    ui.label("r");
                    ui.add(egui::Slider::new(&mut pos.r, 0.0..=50.0));

                    ui.label("phi");
                    ui.add(egui::Slider::new(
                            &mut pos.phi,
                            -PI..=PI,
                    ));

                    ui.label("theta");
                    ui.add(egui::Slider::new(
                            &mut pos.theta,
                            0.0..=2.0 * PI
                    ));


                    ui.checkbox(&mut self.enable_lighting, "lighting");


                    ui.separator();

                    ui.label("values");

                    for v in self.values.iter_mut() {
                        ui.add(egui::Slider::new(
                                v,
                                0.0..=1.0
                        ));
                    }

                    ui.label("bools");

                    for v in self.bools.iter_mut() {
                        ui.checkbox(v, "");
                    }
                });
            });

            ctx.swap_buffers();
        }

        *control_flow = if quit {
            glutin::event_loop::ControlFlow::Exit
        } else {
            glutin::event_loop::ControlFlow::WaitUntil(
                next_draw,
            )
        };
    }
}


fn bellish(x : f32) -> f32 {
    let ex = (-x).exp();

    4.0 * ex / (1.0 + ex).powi(2)
}

fn main() {
    run::<Demo1>();
}

