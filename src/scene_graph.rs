use glam::Mat4;
use glam::Vec3;
use std::cell::RefCell;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn lights_1() {
        run_test_lights(|g| {
            let n1 = g.add_light(1);

            (
                n1,
                vec![
                    (Mat4::IDENTITY, 1)
                ],
            )
        });
    }

    #[test]
    fn lights_2() {
        run_test_lights(|g| {

            let scale = Mat4::from_scale(Vec3::new(2.0, 2.0, 2.0));
            let trans1 = Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0));
            let trans2 = Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0));

            let l1 = g.add_light(1);

            let b = g.add_branch(&[
                (
                    scale,
                    l1,
                ),
            ]);

            let l2 = g.add_light(2);

            let b = g.add_branch(&[
                (
                    trans1,
                    b
                ),
                // note that even thought n2 gets put in the middle of this branch,
                // the resulting lights are grouped by the origin
                (
                    trans2,
                    l2
                ),
                (
                    trans2,
                    b
                ),
            ]);

            // this should get ignored since it's not connected to the root
            let _b = g.add_branch(&[
                (
                    scale,
                    l1,
                ),
            ]);

            (
                b,
                vec![
                    (trans1 * scale, 1),
                    (trans2 * scale, 1),
                    (trans2, 2),
                ],
            )
        });
    }

    fn run_test_lights<F>(f : F)
    where
        F : FnOnce(&mut SceneGraph<i32, (), ()>) -> (NodeHandle, Vec<(Mat4, i32)>)
    {
        let mut g = SceneGraph::<i32, (), ()>::default();
        let (root, expect) = f(&mut g);

        let mut visited = Vec::new();

        g.visit_lights(
            &mut Default::default(),
            root,
            &mut |model, params| {
                visited.push((model, *params));
            }
        );

        assert_eq!(
            visited,
            expect,
        );
    }

    #[test]
    fn surfaces_1() {
        run_test_surfaces(|g| {
            let o = g.add_object(-1);

            let n = g.add_surface(1, o);

            (
                n,
                vec![
                    (Mat4::IDENTITY, 1, -1),
                ],
            )
        });
    }

    #[test]
    fn surfaces_2() {
        run_test_surfaces(|g| {
            let scale = Mat4::from_scale(Vec3::new(2.0, 2.0, 2.0));
            let trans1 = Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0));

            let o1 = g.add_object(-1);
            let o2 = g.add_object(-2);

            let n11 = g.add_surface(1, o1);
            let n12 = g.add_surface(2, o1);

            let n21 = g.add_surface(1, o2);
            let n22 = g.add_surface(2, o2);

            let n = g.add_branch(&[
                (
                    scale,
                    n11,
                ),
                (
                    scale,
                    n21,
                ),
                (
                    trans1,
                    n22,
                ),
                (
                    scale,
                    n12,
                ),
                (
                    scale,
                    n22,
                ),
            ]);

            (
                n,
                vec![
                    (scale, 1, -1),
                    (scale, 2, -1),
                    (scale, 1, -2),
                    (trans1, 2, -2),
                    (scale, 2, -2),
                ],
            )
        });
    }

    #[test]
    fn surfaces_3() {
        run_test_surfaces(|g| {
            let scale = Mat4::from_scale(Vec3::new(2.0, 2.0, 2.0));
            let trans1 = Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0));
            let trans2 = Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0));

            let o = g.add_object(-1);

            let s = g.add_surface(1, o);

            let b = g.add_branch(&[
                (
                    scale,
                    s,
                ),
            ]);

            let b = g.add_branch(&[
                (
                    trans1,
                    b,
                ),
                (
                    trans2,
                    b,
                ),
            ]);

            // this should get ignored since it's not connected to the root
            let _b = g.add_branch(&[
                (
                    scale,
                    s,
                ),
            ]);

            (
                b,
                vec![
                    (trans1 * scale, 1, -1),
                    (trans2 * scale, 1, -1),
                ],
            )
        });
    }

    fn run_test_surfaces<F>(f : F)
    where
        F : FnOnce(&mut SceneGraph<(), i32, i32>) -> (NodeHandle, Vec<(Mat4, i32, i32)>)
    {
        let mut g = SceneGraph::<(), i32, i32>::default();
        let (root, expect) = f(&mut g);

        let mut visited = Vec::new();

        g.visit_surfaces(
            &mut Default::default(),
            root,
            &mut |model, params, object| {
                visited.push((model, *params, *object));
            }
        );

        assert_eq!(
            visited,
            expect
        );
    }

    #[test]
    fn cache() {
        let mut g = SceneGraph::<(), i32, i32>::default();

        let scale = Mat4::from_scale(Vec3::new(2.0, 2.0, 2.0));
        let trans1 = Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0));
        let trans2 = Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0));

        let o = g.add_object(-1);

        let s1 = g.add_surface(1, o);
        let s2 = g.add_surface(2, o);

        let b = g.add_branch(&[
            (
                scale,
                s1,
            ),
        ]);

        let b = g.add_branch(&[
            (
                trans1,
                b,
            ),
            (
                trans2,
                b,
            ),
        ]);

        // this should get ignored since it's not connected to the root
        let _b = g.add_branch(&[
            (
                scale,
                s2,
            ),
        ]);

        let mut cache = Cache::default();
        g.fill_cache(&mut cache, b);

        assert_eq!(cache.root, Some(b));
        assert_eq!(
            cache.nodes,
            vec![
                true,
                false,
                true,
                true,
                false,
            ].into_iter().map(Some).collect::<Vec<_>>(),
        );
    }
}



enum Node<L, S> {
    Light{
        params : L,
        parents : Vec<(Mat4, NodeHandle)>,
    },
    Surface{
        params : S,
        parents : Vec<(Mat4, NodeHandle)>,
    },
    Branch{
        parents : Vec<(Mat4, NodeHandle)>,
    },
}

impl<L, S> Node<L, S> {
    fn parents_mut(&mut self) -> &mut Vec<(Mat4, NodeHandle)> {
        use Node::*;

        match self {
            Light{parents, ..} => parents,
            Surface{parents, ..} => parents,
            Branch{parents, ..} => parents,
        }
    }

    fn parents(&self) -> &[(Mat4, NodeHandle)] {
        use Node::*;

        match self {
            Light{parents, ..} => parents,
            Surface{parents, ..} => parents,
            Branch{parents, ..} => parents,
        }
    }
}

struct Object<O> {
    object : O,
    parents : Vec<NodeHandle>,
}

#[derive(Clone, Copy)]
pub struct ObjectHandle(usize);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NodeHandle(usize);

pub struct SceneGraph<L, S, O> {
    objects : Vec<Object<O>>,
    nodes : Vec<Node<L, S>>,
}

impl<L, S, O> Default for SceneGraph<L, S, O> {
    fn default() -> Self {
        Self{
            objects : Vec::new(),
            nodes : Vec::new(),
        }
    }
}

#[derive(Default)]
pub struct Cache {
    root : Option<NodeHandle>,
    nodes : Vec<Option<bool>>,
}

impl<L, S, O> SceneGraph<L, S, O> {

    fn fill_cache_rec(
        &self,
        cache : &mut Cache,
        root : NodeHandle,
        cur : NodeHandle,
    ) -> bool {
        if cur == root {
            cache.nodes[cur.0] = Some(true);
            return true
        }

        if let Some(cached) = cache.nodes[cur.0] {
            return cached
        }

        for (_, parent) in self.nodes[cur.0].parents() {
            if cache.nodes[parent.0] == Some(true) || self.fill_cache_rec(cache, root, *parent) {
                cache.nodes[cur.0] = Some(true);
                return true
            }
        }

        cache.nodes[cur.0] = Some(false);
        false
    }

    fn fill_cache(
        &self,
        cache : &mut Cache,
        root : NodeHandle,
    ) {
        if cache.root != Some(root) ||
            cache.nodes.len() != self.nodes.len()
        {
            cache.root = Some(root);
            cache.nodes.clear(); // TODO: is this necessary?
            cache.nodes.resize_with(self.nodes.len(), || None);
        }

        for idx in 0..self.nodes.len() {
            self.fill_cache_rec(
                cache,
                root,
                NodeHandle(idx),
            );
        }
    }

    fn visit_lights_rec<F>(
        &self,
        cache : &mut Cache,
        root : NodeHandle,
        params : &L,
        cur : NodeHandle,
        trans : Mat4,
        f : &mut F
    )
    where
        F : FnMut(Mat4, &L)
    {
        if root == cur {
            f(trans, params);
            return
        } else if cache.nodes[cur.0].unwrap() {
            for (trans_next, next) in self.nodes[cur.0].parents().iter().copied() {
                self.visit_lights_rec(
                    cache,
                    root,
                    params,
                    next,
                    trans_next * trans,
                    f,
                );
            }
        }
    }

    pub fn visit_lights<F>(
        &self,
        cache : &mut Cache,
        root : NodeHandle,
        f : &mut F,
    )
    where
        F : FnMut(Mat4, &L)
    {
        self.fill_cache(cache, root);

        for (idx, node) in self.nodes.iter().enumerate() {
            if let Node::Light{params,..} = node {
                self.visit_lights_rec(
                    cache,
                    root,
                    params,
                    NodeHandle(idx),
                    Mat4::IDENTITY,
                    f,
                );
            }
        }
    }

    fn visit_surfaces_rec<F>(
        &self,
        cache : &mut Cache,
        root : NodeHandle,
        params : &S,
        object : &O,
        cur : NodeHandle,
        trans : Mat4,
        f : &mut F,
    )
    where
        F : FnMut(Mat4, &S, &O)
    {
        if root == cur {
            f(trans, params, object);
            return
        } else if cache.nodes[cur.0].unwrap() {
            for (trans_next, next) in self.nodes[cur.0].parents().iter().copied() {
                self.visit_surfaces_rec(
                    cache,
                    root,
                    params,
                    object,
                    next,
                    trans_next * trans,
                    f,
                );
            }
        }
    }

    pub fn visit_surfaces<F>(
        &self,
        cache : &mut Cache,
        root : NodeHandle,
        f : &mut F,
    )
    where
        F : FnMut(Mat4, &S, &O)
    {
        self.fill_cache(cache, root);

        for Object{object, parents} in self.objects.iter() {
            for handle in parents.iter().copied() {
                if let Node::Surface{params, ..} = &self.nodes[handle.0] {
                    self.visit_surfaces_rec(
                        cache,
                        root,
                        params,
                        object,
                        handle,
                        Mat4::IDENTITY,
                        f
                    );
                } else {
                    unreachable!("node pointed by Object.parent should be surface");
                }
            }
        }
    }

    pub fn add_object(&mut self, object : O) -> ObjectHandle {
        self.objects.push(Object{
            object,
            parents : Vec::new(),
        });

        ObjectHandle(self.objects.len() - 1)
    }

    fn add_node(&mut self, node : Node<L, S>) -> NodeHandle {
        self.nodes.push(node);
        NodeHandle(self.nodes.len() - 1)
    }

    pub fn add_surface(&mut self, params : S, object : ObjectHandle) -> NodeHandle {
        let handle = self.add_node(Node::Surface{
            params,
            parents : Vec::new(),
        });

        self.objects[object.0].parents.push(handle);

        handle
    }

    pub fn add_light(&mut self, params : L) -> NodeHandle {
        self.add_node(Node::Light{
            params,
            parents : Vec::new(),
        })
    }

    pub fn add_branch(&mut self, children : &[(Mat4, NodeHandle)]) -> NodeHandle {
        let handle = self.add_node(Node::Branch{
            parents : Vec::new(),
        });

        for (trans, child) in children.iter().copied() {
            self.nodes[child.0].parents_mut().push((trans, handle));
        }

        handle
    }
}
