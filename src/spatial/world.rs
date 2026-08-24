use rapier3d::prelude::*;

pub struct PhysicsWorld {
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
    ball_handle: Option<RigidBodyHandle>,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        let gravity = vector![0.0, -9.81, 0.0];
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

        let ground =
            ColliderBuilder::halfspace(rapier3d::na::Unit::new_normalize(vector![0.0, 1.0, 0.0]))
                .build();
        colliders.insert(ground);

        let ball_sz = RigidBodyBuilder::dynamic()
            .translation(vector![0.0, 5.0, 0.0])
            .build();
        let ball_h = bodies.insert(ball_sz);
        let ball_c = ColliderBuilder::ball(0.5).build();
        colliders.insert_with_parent(ball_c, ball_h, &mut bodies);

        let cube_sz = RigidBodyBuilder::fixed()
            .translation(vector![3.0, 0.5, 0.0])
            .build();
        let cube_h = bodies.insert(cube_sz);
        let cube_c = ColliderBuilder::cuboid(0.5, 0.5, 0.5).build();
        colliders.insert_with_parent(cube_c, cube_h, &mut bodies);

        let wall_sz = RigidBodyBuilder::fixed()
            .translation(vector![-3.0, 1.0, 0.0])
            .build();
        let wall_h = bodies.insert(wall_sz);
        let wall_c = ColliderBuilder::cuboid(0.2, 1.0, 2.0).build();
        colliders.insert_with_parent(wall_c, wall_h, &mut bodies);

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
            ball_handle: Some(ball_h),
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

    pub fn ray_cast(
        &self,
        origin: &[f32; 3],
        dir: &[f32; 3],
        max_dist: f32,
    ) -> (f32, Option<String>) {
        let offset_origin = [origin[0], origin[1] + 0.6, origin[2]];
        let ray = Ray::new(
            point![offset_origin[0], offset_origin[1], offset_origin[2]],
            vector![dir[0], dir[1], dir[2]],
        );
        let filter = QueryFilter::default().exclude_sensors();
        if let Some((_handle, toi)) = self.query_pipeline.cast_ray(
            &self.bodies,
            &self.colliders,
            &ray,
            max_dist,
            false,
            filter,
        ) {
            (toi, Some(format!("hit_{}", toi)))
        } else {
            (max_dist, None)
        }
    }

    pub fn camera_pos(&self) -> [f64; 3] {
        if let Some(h) = self.ball_handle {
            let p = self.bodies[h].translation();
            [p.x as f64, p.y as f64, p.z as f64]
        } else {
            [0.0; 3]
        }
    }

    pub fn camera_vel(&self) -> [f64; 3] {
        if let Some(h) = self.ball_handle {
            let v = self.bodies[h].linvel();
            [v.x as f64, v.y as f64, v.z as f64]
        } else {
            [0.0; 3]
        }
    }

    pub fn apply_force(&mut self, force: &[f32; 3]) {
        if let Some(h) = self.ball_handle {
            self.bodies[h].add_force(vector![force[0], force[1], force[2]], true);
        }
    }
}
