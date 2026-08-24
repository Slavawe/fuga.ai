use rapier3d::prelude::*;

pub struct Room {
    pub gravity: Vector<f32>,
    pub integration_parameters: IntegrationParameters,
    pub physics_pipeline: PhysicsPipeline,
    pub island_manager: IslandManager,
    pub broad_phase: BroadPhase,
    pub narrow_phase: NarrowPhase,
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub impulse_joints: ImpulseJointSet,
    pub multibody_joints: MultibodyJointSet,
    pub query_pipeline: QueryPipeline,
    pub ccd_solver: CCDSolver,
    sphere_handle: Option<RigidBodyHandle>,
    sphere_collider: Option<ColliderHandle>,
    pub room_size: f32,
}

impl Room {
    pub fn new(half_extent: f32) -> Self {
        let gravity = vector![0.0, 0.0, 0.0];
        let integration_parameters = IntegrationParameters::default();
        let physics_pipeline = PhysicsPipeline::new();
        let island_manager = IslandManager::new();
        let broad_phase = BroadPhase::new();
        let narrow_phase = NarrowPhase::new();
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let impulse_joints = ImpulseJointSet::new();
        let multibody_joints = MultibodyJointSet::new();
        let query_pipeline = QueryPipeline::new();
        let ccd_solver = CCDSolver::new();
        let he = half_extent;

        let normal_up = rapier3d::na::Unit::new_normalize(vector![0.0, 1.0, 0.0]);
        let normal_down = rapier3d::na::Unit::new_normalize(vector![0.0, -1.0, 0.0]);
        let normal_x = rapier3d::na::Unit::new_normalize(vector![1.0, 0.0, 0.0]);
        let normal_neg_x = rapier3d::na::Unit::new_normalize(vector![-1.0, 0.0, 0.0]);
        let normal_z = rapier3d::na::Unit::new_normalize(vector![0.0, 0.0, 1.0]);
        let normal_neg_z = rapier3d::na::Unit::new_normalize(vector![0.0, 0.0, -1.0]);

        let floor = ColliderBuilder::halfspace(normal_up)
            .translation(vector![0.0, -he, 0.0])
            .build();
        colliders.insert(floor);

        let ceiling = ColliderBuilder::halfspace(normal_down)
            .translation(vector![0.0, he, 0.0])
            .build();
        colliders.insert(ceiling);

        let wall_nx = ColliderBuilder::halfspace(normal_x)
            .translation(vector![-he, 0.0, 0.0])
            .build();
        colliders.insert(wall_nx);

        let wall_px = ColliderBuilder::halfspace(normal_neg_x)
            .translation(vector![he, 0.0, 0.0])
            .build();
        colliders.insert(wall_px);

        let wall_nz = ColliderBuilder::halfspace(normal_z)
            .translation(vector![0.0, 0.0, -he])
            .build();
        colliders.insert(wall_nz);

        let wall_pz = ColliderBuilder::halfspace(normal_neg_z)
            .translation(vector![0.0, 0.0, he])
            .build();
        colliders.insert(wall_pz);

        let sphere = RigidBodyBuilder::dynamic()
            .translation(vector![0.0, 0.0, 0.0])
            .build();
        let sphere_h = bodies.insert(sphere);
        let sphere_c = ColliderBuilder::ball(0.3).build();
        let sphere_ch = colliders.insert_with_parent(sphere_c, sphere_h, &mut bodies);

        Self {
            gravity,
            integration_parameters,
            physics_pipeline,
            island_manager,
            broad_phase,
            narrow_phase,
            bodies,
            colliders,
            impulse_joints,
            multibody_joints,
            query_pipeline,
            ccd_solver,
            sphere_handle: Some(sphere_h),
            sphere_collider: Some(sphere_ch),
            room_size: half_extent,
        }
    }

    pub fn step(&mut self, dt: f64) {
        self.integration_parameters.dt = dt as f32;
        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &(),
            &(),
        );
    }

    pub fn sphere_pos(&self) -> [f64; 3] {
        if let Some(h) = self.sphere_handle {
            let p = self.bodies[h].translation();
            [p.x as f64, p.y as f64, p.z as f64]
        } else {
            [0.0; 3]
        }
    }

    pub fn sphere_vel(&self) -> [f64; 3] {
        if let Some(h) = self.sphere_handle {
            let v = self.bodies[h].linvel();
            [v.x as f64, v.y as f64, v.z as f64]
        } else {
            [0.0; 3]
        }
    }

    pub fn cast_ray(&self, origin: &[f32; 3], dir: &[f32; 3], max_dist: f32) -> f32 {
        let ray = Ray::new(
            point![origin[0], origin[1], origin[2]],
            vector![dir[0], dir[1], dir[2]],
        );
        let filter = QueryFilter::default()
            .exclude_rigid_body(self.sphere_handle.expect("sphere exists"))
            .exclude_sensors();
        if let Some((_handle, toi)) = self.query_pipeline.cast_ray(
            &self.bodies,
            &self.colliders,
            &ray,
            max_dist,
            false,
            filter,
        ) {
            toi
        } else {
            max_dist
        }
    }

    pub fn apply_force(&mut self, force: &[f32; 3]) {
        if let Some(h) = self.sphere_handle {
            self.bodies[h].add_force(vector![force[0], force[1], force[2]], true);
        }
    }
}
