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
    SceneGraph,
    Node,
};

use std::io;
use std::fs::File;
use std::f32::consts::PI;

use winit_input_helper::WinitInputHelper;

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



struct MyScene {
    graph : SceneGraph<Light, Surface, LoadedObj>,
    // cache : scene_graph::Cache,

    sphere : NodeHandle,
    cube : NodeHandle,
    bunny : NodeHandle,
    teapot : NodeHandle,
}

impl MyScene {
    fn new(ctx : &mut GraphicsContext) -> Self {
        let f = File::open("models/teapot.obj").unwrap();
        let obj = Obj::parse(io::BufReader::new(f)).unwrap();
        let teapot = ctx.load_object(&obj);

        let f = File::open("models/sphere.obj").unwrap();
        let obj = Obj::parse(io::BufReader::new(f)).unwrap();
        let sphere = ctx.load_object(&obj);

        let f = File::open("models/bunny.obj").unwrap();
        let obj = Obj::parse(io::BufReader::new(f)).unwrap();
        let bunny = ctx.load_object(&obj);

        let cube = ctx.load_object(&Obj::cube());

        let mut g = scene_graph::SceneGraph::<
            Light,
            Surface,
            LoadedObj
        >::default();

        fn add_object_surface(
            g : &mut scene_graph::SceneGraph::<Light, Surface, LoadedObj>,
            object : LoadedObj,
            surface : Surface,
        ) -> scene_graph::NodeHandle {
            let o = g.add_object(object);
            g.add_surface(surface, o)
        }

        let sphere = add_object_surface(
            &mut g,
            sphere,
            Surface{ material : PhongMaterial::turquoise() },
        );

        let teapot = add_object_surface(
            &mut g,
            teapot,
            Surface{ material : PhongMaterial::turquoise() },
        );

        let cube = add_object_surface(
            &mut g,
            cube,
            Surface{ material : PhongMaterial::turquoise() },
        );

        let bunny = add_object_surface(
            &mut g,
            bunny,
            Surface{ material : PhongMaterial::turquoise() },
        );

        Self {
            graph : g,
            // cache : Default::default(),
            sphere,
            cube,
            bunny,
            teapot,
        }
    }
}



struct MyApp {
    values : [f32; 3],
    bools : [bool; 3],

    enable_lighting : bool,

    prog : LoadedProg,

    camera : Camera,
    camera_pos : SphereCoord,

    input : WinitInputHelper,

    render_cache : RenderCache,

    shadow_map_debug: TextureDebug,

    scene : MyScene,
}

impl App for MyApp {
    fn init(ctx : &mut GraphicsContext) -> Self {
        // try frame buffer

        ctx.set_title("game-engine");

        let (major, minor) = ctx.gl_version();
        println!("OpenGL version: {}.{}", major, minor);

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

        let mut grad = image::RgbImage::new(32, 32);
        for x in 0u8..32 {
            for y in 0u8..32 {
                grad.put_pixel(
                    x as u32,
                    y as u32,
                    image::Rgb([x * 8, y * 8, 0])
                );
            }
        }

        Self{
            values: [0.0;3],
            bools: [false;3],
            enable_lighting : true,
            prog,
            camera,
            camera_pos : SphereCoord {
                r : 5.0,
                theta: 0.0,
                phi : 0.0,
            },
            input : WinitInputHelper::new(),
            render_cache : RenderCache::new(ctx),
            shadow_map_debug,
            scene : MyScene::new(ctx),
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

        if is_redraw_event(&event) {
            ctx.clear(0.1, 0.1, 0.1, 1.0);

            let bulb_mat = Mat4::from_translation(Vec3::new(
                    0.0,
                    3.0,
                    3.0,
            ));

            let bulb_light = Node::Light(
                Light::Point{
                    color : if self.bools[0] {
                        1.0
                    } else {
                        0.0
                    } * Vec4::new(1.0, 1.0, 1.0, 1.0),
                    shadow : false,
                },
            );

            /*
            let light_idx = g.add_branch(&[
                (
                    /*
                    if self.bools[0] {
                        // put light at infinity
                        Mat4::from_cols_array_2d(&[
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 1.0, 0.0],
                        ])
                    } else {
                        Mat4::IDENTITY
                    },
                    */
                    Mat4::IDENTITY,
                    l,
                ),
                (
                    Mat4::IDENTITY,
                    cube_idx,
                ),
            ]);
                    */

            let spotlight = Node::Light(if self.bools[2] {
                Light::Directional{
                    color : Vec4::new(1.0, 0.0, 1.0, 1.0),
                    shadow : true,
                    left : -1.0, right : 1.0,
                    bottom : -1.0, top : 1.0,
                    near : 0.1, far : 10.0,
                }
            } else {
                Light::Spot{
                    color : Vec4::new(1.0, 0.0, 1.0, 1.0),
                    forward : -Vec3::Z,
                    shadow : true,
                    angle : PI / 2.0,
                    near : 1.0,
                    far : 10.0,
                }
            });

            let roots = &[
                (
                    Mat4::from_translation(Vec3::new(0.8, 0.0, 0.0)),
                    self.scene.bunny.into(),
                ),
                (
                    Mat4::from_translation(Vec3::new(
                            -0.8,
                            0.1 + 2.0 * self.values[2],
                            -4.0
                    )),
                    self.scene.teapot.into(),
                ),
                (
                    Mat4::from_scale_rotation_translation(
                        Vec3::new(4.0, 0.1, 4.0),
                        glam::Quat::IDENTITY,
                        Vec3::new(0.0, -1.5, 0.0)
                    ),
                    self.scene.sphere.into(),
                ),
                (
                    bulb_mat,
                    bulb_light,
                ),
                (
                    bulb_mat,
                    self.scene.cube.into(),
                ),
                (
                    Mat4::from_translation(
                        Vec3::new(
                            0.0,
                            4.0 * (self.values[0] - 0.5),
                            10.0 * (self.values[1] - 0.5)
                        ),
                    ),
                    spotlight,
                )
            ];

            self.camera.aspect = ctx.aspect();
            self.camera.position = self.camera_pos.position();
            self.camera.up = self.camera_pos.up();


            ctx.render_scene(
                &mut self.render_cache,
                &self.camera,
                &scene_graph::Scene::new(
                    &self.scene.graph,
                    roots,
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
                egui::SidePanel::left("left_panel")
                .resizable(true)
                .show(ctx, |ui| {
                    ui.heading("hello world");
                    quit = ui.button("Quit").clicked();


                    let pos = &mut self.camera_pos;

                    ui.label("r");
                    ui.add(egui::Slider::new(&mut pos.r, 0.0..=20.0));

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
            glutin::event_loop::ControlFlow::Wait
        };
    }
}


fn main() {
    run::<MyApp>();
}

