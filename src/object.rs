use std::io;
use std::str::FromStr;

use glam::{
    Vec3,
};

#[derive(Debug)]
pub enum Error {
    ObjFormat(usize),
}

#[derive(Clone, Copy)]
pub struct LoadedObj {
    pub(crate) vao : glow::NativeVertexArray,
    pub(crate) count : usize,
    pub(crate) index_vbo : glow::NativeBuffer,
    pub(crate) vertex_vbo : glow::NativeBuffer,
    pub(crate) normal_vbo : glow::NativeBuffer,
}

pub struct Obj {
    pub indices : Vec<u32>,
    pub vertices : Vec<Vec3>,
    pub normals : Vec<Vec3>,
}

impl Obj {
    pub fn plane() -> Self {

        use glam::{
            Vec3,
            const_vec3,
        };

        const VERTICES : [Vec3;4] = [
            // Front face
            const_vec3!([ -0.5, -0.5, 0.0 ]),
            const_vec3!([ -0.5,  0.5, 0.0 ]),
            const_vec3!([  0.5,  0.5, 0.0 ]),
            const_vec3!([  0.5, -0.5, 0.0 ]),
        ];

        const NORMALS : [Vec3;4] = [
            // Front face
            const_vec3!([ 0.0, 0.0, 1.0 ]),
            const_vec3!([ 0.0, 0.0, 1.0 ]),
            const_vec3!([ 0.0, 0.0, 1.0 ]),
            const_vec3!([ 0.0, 0.0, 1.0 ]),
        ];

        const INDICES : [u32;6] = [
            0, 1, 2, 0, 2, 3, // Front face
        ];

        Obj{
            indices : INDICES.to_vec(),
            vertices : VERTICES.to_vec(),
            normals : NORMALS.to_vec(),
        }
    }

    pub fn cube() -> Self {

        use glam::{
            Vec3,
            const_vec3,
        };

        const VERTICES : [Vec3;24] = [
            // Front face
            const_vec3!([ -0.5, -0.5, 0.5 ]),
            const_vec3!([ -0.5, 0.5, 0.5 ]),
            const_vec3!([ 0.5, 0.5, 0.5 ]),
            const_vec3!([ 0.5, -0.5, 0.5 ]),

            // Back face
            const_vec3!([ -0.5, -0.5, -0.5 ]),
            const_vec3!([ -0.5, 0.5, -0.5 ]),
            const_vec3!([ 0.5, 0.5, -0.5 ]),
            const_vec3!([ 0.5, -0.5, -0.5 ]),

            // Left face
            const_vec3!([ -0.5, -0.5, 0.5 ]),
            const_vec3!([ -0.5, 0.5, 0.5 ]),
            const_vec3!([ -0.5, 0.5, -0.5 ]),
            const_vec3!([ -0.5, -0.5, -0.5 ]),

            // Right face
            const_vec3!([ 0.5, -0.5, 0.5 ]),
            const_vec3!([ 0.5, 0.5, 0.5 ]),
            const_vec3!([ 0.5, 0.5, -0.5 ]),
            const_vec3!([ 0.5, -0.5, -0.5 ]),

            // Top face
            const_vec3!([ 0.5, 0.5, 0.5 ]),
            const_vec3!([ -0.5, 0.5, 0.5 ]),
            const_vec3!([ -0.5, 0.5, -0.5 ]),
            const_vec3!([ 0.5, 0.5, -0.5 ]),

            // Bottom face
            const_vec3!([ 0.5, -0.5, 0.5 ]),
            const_vec3!([ -0.5, -0.5, 0.5 ]),
            const_vec3!([ -0.5, -0.5, -0.5 ]),
            const_vec3!([ 0.5, -0.5, -0.5 ])
        ];

        const NORMALS : [Vec3;24] = [
            // Front face
            const_vec3!([0.0, 0.0, 1.0]),
            const_vec3!([0.0, 0.0, 1.0]),
            const_vec3!([ 0.0, 0.0, 1.0 ]),
            const_vec3!([ 0.0, 0.0, 1.0 ]),
            // Back face
            const_vec3!([ 0.0, 0.0, -1.0 ]),
            const_vec3!([ 0.0, 0.0, -1.0 ]),
            const_vec3!([ 0.0, 0.0, -1.0 ]),
            const_vec3!([ 0.0, 0.0, -1.0 ]),

            // Left face
            const_vec3!([ -1.0, 0.0, 0.0 ]),
            const_vec3!([ -1.0, 0.0, 0.0 ]),
            const_vec3!([ -1.0, 0.0, 0.0 ]),
            const_vec3!([ -1.0, 0.0, 0.0 ]),

            // Right face
            const_vec3!([ 1.0, 0.0, 0.0 ]),
            const_vec3!([ 1.0, 0.0, 0.0 ]),
            const_vec3!([ 1.0, 0.0, 0.0 ]),
            const_vec3!([ 1.0, 0.0, 0.0 ]),

            // Top face
            const_vec3!([ 0.0, 1.0, 0.0 ]),
            const_vec3!([ 0.0, 1.0, 0.0 ]),
            const_vec3!([ 0.0, 1.0, 0.0 ]),
            const_vec3!([ 0.0, 1.0, 0.0 ]),

            // Bottom face
            const_vec3!([ 0.0, -1.0, 0.0 ]),
            const_vec3!([ 0.0, -1.0, 0.0 ]),
            const_vec3!([ 0.0, -1.0, 0.0 ]),
            const_vec3!([ 0.0, -1.0, 0.0 ]),
        ];

        const INDICES : [u32;36] = [
            0, 1, 2, 0, 2, 3, // Front face
            4, 5, 6, 4, 6, 7, // Back face
            8, 9, 10, 8, 10, 11, // Left face
            12, 13, 14, 12, 14, 15, // Right face
            16, 17, 18, 16, 18, 19, // Top face
            20, 21, 22, 20, 22, 23 // Bottom face
        ];

        Obj{
            indices : INDICES.to_vec(),
            vertices : VERTICES.to_vec(),
            normals : NORMALS.to_vec(),
        }
    }

    pub fn parse<R : io::BufRead> (mut r : R) -> Result<Obj, Error> {
        fn extract<'a, T, I>(
            line_num: usize,
            mut it : I,
            dst : &mut [T]
        ) -> Result<(), Error>
        where
            T : FromStr,
            <T as FromStr>::Err : std::fmt::Debug,
            I : Iterator<Item = &'a str>
        {
            for el in dst.iter_mut() {
                let s = it.next().ok_or(Error::ObjFormat(line_num))?;

                match FromStr::from_str(s) {
                    Ok(v) => {
                        *el = v;
                    },
                    Err(_) => {
                        return Err(Error::ObjFormat(line_num))
                    }
                }
            }

            Ok(())
        }

        let mut vertices = Vec::new();
        let mut normals = Vec::new();
        let mut vertex_indices = Vec::new();
        let mut normal_indices = Vec::new();


        let mut buf = String::new();
        let mut line_num = 0;

        // TODO: handle errors from readline
        while let Ok(n) = r.read_line(&mut buf) {
            if n == 0 {
                break;
            }

            line_num += 1;

            let mut it = buf.trim_end().split(' ');

            match it.next() {
                Some("v") => {
                    let mut vertex = Vec3::ZERO;
                    extract(line_num, it, vertex.as_mut())?;
                    vertices.push(vertex);
                },
                Some("vn") => {
                    let mut normal = Vec3::ZERO;
                    extract(line_num, it, normal.as_mut())?;
                    normals.push(normal);
                },
                Some("f") => {

                    let pairs = it
                        .take(3)
                        .map(|s| {
                            s.split_once("//").map(|(x, y)| [x, y])
                        })
                        .take_while(Option::is_some)
                        .flat_map(Option::unwrap);

                    let mut nums = [0usize; 6];
                    extract(line_num, pairs, &mut nums)?;
                    for i in 0..3 {
                        vertex_indices.push(nums[2*i]);
                        normal_indices.push(nums[2*i+1]);
                    }
                },
                Some("g" | "#" | "\n" | "") => {}, // ignore for now
                Some(s) if s.starts_with('#') => {},
                s => panic!("bad format: {:?}", s),
            }

            buf.clear();
        }

        let n = vertex_indices.len();

        let mut ret = Obj{
            indices : Vec::with_capacity(n),
            vertices : Vec::with_capacity(n),
            normals : Vec::with_capacity(n),
        };

        for i in 0..n {
            ret.indices.push(i as u32);
            ret.vertices.push(vertices[vertex_indices[i] - 1]);
            ret.normals.push(normals[normal_indices[i] - 1]);
        }

        Ok(ret)
    }
}
