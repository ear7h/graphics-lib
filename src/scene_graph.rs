use glam::Mat4;

#[cfg(test)]
mod test {
    use super::*;
    use glam::Vec3;

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

        Scene::new(
            &mut g,
            &[(Mat4::IDENTITY, Node::Handle(root))],
            &mut Default::default(),
        ).visit_lights(
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

        Scene::new(
            &mut g,
            &[(Mat4::IDENTITY, Node::Handle(root))],
            &mut Default::default(),
        ).visit_surfaces(
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

        let b1 = g.add_branch(&[
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

        let b2 = g.add_branch(&[
            (
                scale,
                b1,
            ),
        ]);

        let mut cache = Cache::default();
        Scene::new(
            &mut g,
            &[
                (Mat4::IDENTITY, Node::Handle(b1)),
                (Mat4::IDENTITY, Node::Handle(b2)),
            ],
            &mut cache,
        ).fill_cache();

        use Reach::*;

        assert_eq!(
            cache.nodes,
            vec![
                Reached,
                None,
                Reached,
                RootReached,
                None,
                Root,
            ].into_iter().map(Some).collect::<Vec<_>>(),
        );
    }
}


#[derive(Debug, Clone, Copy)]
pub enum Node<L, S> {
    Light(L),
    Surface(ObjectHandle, S),
    Handle(NodeHandle),
}

impl<L, S> From<NodeHandle> for Node<L, S> {
    fn from(handle : NodeHandle) -> Node<L, S> {
        Node::Handle(handle)
    }
}

enum NodeInternal<L, S> {
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

impl<L, S> NodeInternal<L, S> {
    fn parents_mut(&mut self) -> &mut Vec<(Mat4, NodeHandle)> {
        use NodeInternal::*;

        match self {
            Light{parents, ..} => parents,
            Surface{parents, ..} => parents,
            Branch{parents, ..} => parents,
        }
    }

    fn parents(&self) -> &[(Mat4, NodeHandle)] {
        use NodeInternal::*;

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

#[derive(Debug, Clone, Copy)]
pub struct ObjectHandle(usize);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NodeHandle(usize);

pub struct Scene<'a, L, S, O> {
    scene_graph : &'a SceneGraph<L, S, O>,
    roots : &'a [(Mat4, Node<L, S>)],
    cache : &'a mut Cache,
}

impl<'a, L, S, O> Scene<'a, L, S, O> {
    pub fn new(
        scene_graph: &'a SceneGraph<L, S, O> ,
        roots : &'a [(Mat4, Node<L, S>)],
        cache : &'a mut Cache,
    ) -> Self {
        let mut ret = Self{ scene_graph, roots, cache };
        ret.fill_cache();
        ret
    }

    fn fill_cache_rec(
        &mut self,
        cur : NodeHandle,
    ) -> Reach {
        if let Some(cached) = self.cache.nodes[cur.0] {
            return cached
        }

        let mut reach = false;

        for (_, parent) in self.scene_graph.nodes[cur.0].parents() {
            use Reach::*;

            let r = if let Some(r) = self.cache.nodes[parent.0] {
                r
            } else {
                self.fill_cache_rec(*parent)
            };

            match r {
                None => continue,
                Reached | RootReached | Root => {
                    reach = true;
                    break;
                }
            };
        }

        let is_root = self.roots.iter().any(|(_, node)| {
            matches!(
                node,
                Node::Handle(handle) if *handle == cur,
            )
        });

        use Reach::*;

        let val = match (reach, is_root) {
            (false, false) => None,
            (true,  false) => Reached,
            (false, true)  => Root,
            (true,  true)  => RootReached,
        };

        self.cache.nodes[cur.0] = Some(val);
        val
    }

    /// This should only be called from `new`
    fn fill_cache(
        &mut self,
    ) {
        self.cache.nodes.clear();
        self.cache.nodes.resize_with(self.scene_graph.nodes.len(), || None);

        for idx in 0..self.scene_graph.nodes.len() {
            self.fill_cache_rec(
                NodeHandle(idx),
            );
        }
    }

    fn visit_surfaces_rec<F>(
        &self,
        params : &S,
        object : &O,
        cur : NodeHandle,
        trans : Mat4,
        f : &mut F,
    )
    where
        F : FnMut(Mat4, &S, &O)
    {
        use Reach::*;

        let r = self.cache.nodes[cur.0]
            .expect("all nodes should have a reach value");

        if matches!(r, Root | RootReached) {
            let it = self.roots
                .iter()
                .filter_map(|(mat, node)| {
                    matches!(
                        node,
                        Node::Handle(handle) if *handle == cur
                    ).then(|| mat)
                });

            for mat in it  {
                f(*mat * trans, params, object);
            }
        }

        if matches!(r, Reached | RootReached) {
            let parents = self.scene_graph.nodes[cur.0]
                .parents()
                .iter()
                .copied();

            for (trans_next, next) in parents {
                self.visit_surfaces_rec(
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
        f : &mut F,
    )
    where
        F : FnMut(Mat4, &S, &O)
    {
        let it = self.scene_graph.objects.iter().enumerate();

        for (idx, Object{object, parents}) in it {
            for (mat, node) in self.roots {
                match node {
                    Node::Surface(o, params) if o.0 == idx => {
                        f(*mat, params, object)
                    },
                    _ => {}
                }
            }

            for handle in parents {
                let node = &self.scene_graph.nodes[handle.0];
                if let NodeInternal::Surface{params, ..} = node {
                    self.visit_surfaces_rec(
                        params,
                        object,
                        *handle,
                        Mat4::IDENTITY,
                        f
                    );
                } else {
                    unreachable!(
                        "node pointed by Object.parent should be surface"
                    );
                }

            }
        }
    }

    fn visit_lights_rec<F>(
        &self,
        params : &L,
        cur : NodeHandle,
        trans : Mat4,
        f : &mut F
    )
    where
        F : FnMut(Mat4, &L)
    {
        use Reach::*;

        let r = self.cache.nodes[cur.0].unwrap();

        if matches!(r, Root | RootReached) {
            let it = self.roots
                .iter()
                .filter_map(|(mat, node)| {
                    matches!(
                        node,
                        Node::Handle(handle) if *handle == cur
                    ).then(|| mat)
                });

            for mat in it {
                f(*mat * trans, params);
            }
        }

        if matches!(r, Reached | RootReached) {
            let parents = self.scene_graph.nodes[cur.0]
                .parents()
                .iter()
                .copied();

            for (trans_next, next) in parents {
                self.visit_lights_rec(
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
        f : &mut F,
    )
    where
        F : FnMut(Mat4, &L)
    {
        for (mat, node) in self.roots {
            if let Node::Light(params) = node {
                f(*mat, params);
            }
        }

        for (idx, node) in self.scene_graph.nodes.iter().enumerate() {
            if let NodeInternal::Light{params,..} = node {
                self.visit_lights_rec(
                    params,
                    NodeHandle(idx),
                    Mat4::IDENTITY,
                    f,
                );
            }
        }
    }
}

pub struct SceneGraph<L, S, O> {
    objects : Vec<Object<O>>,
    nodes : Vec<NodeInternal<L, S>>,
}

impl<L, S, O> Default for SceneGraph<L, S, O> {
    fn default() -> Self {
        Self{
            objects : Vec::new(),
            nodes : Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reach {
    // The node is not reached by a root
    None,
    // The node is reached by a root
    Reached,
    // The node is a root AND IS NOT reached by another root
    Root,
    // The node is a root AND IS reached by another root
    RootReached,
}

#[derive(Default)]
pub struct Cache {
    nodes : Vec<Option<Reach>>,
}

impl<L, S, O> SceneGraph<L, S, O> {

    pub fn add_object(&mut self, object : O) -> ObjectHandle {
        self.objects.push(Object{
            object,
            parents : Vec::new(),
        });

        ObjectHandle(self.objects.len() - 1)
    }

    fn add_node(&mut self, node : NodeInternal<L, S>) -> NodeHandle {
        self.nodes.push(node);
        NodeHandle(self.nodes.len() - 1)
    }

    pub fn add_surface(&mut self, params : S, object : ObjectHandle) -> NodeHandle {
        let handle = self.add_node(NodeInternal::Surface{
            params,
            parents : Vec::new(),
        });

        self.objects[object.0].parents.push(handle);

        handle
    }

    pub fn add_light(&mut self, params : L) -> NodeHandle {
        self.add_node(NodeInternal::Light{
            params,
            parents : Vec::new(),
        })
    }

    pub fn add_branch(&mut self, children : &[(Mat4, NodeHandle)]) -> NodeHandle {
        let handle = self.add_node(NodeInternal::Branch{
            parents : Vec::new(),
        });

        for (trans, child) in children.iter().copied() {
            self.nodes[child.0].parents_mut().push((trans, handle));
        }

        handle
    }

}
