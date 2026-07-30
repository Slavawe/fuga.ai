
#[derive(Clone)]
pub struct Valve {
    pub position: f64,
    pub max_flow: f64,
}

impl Valve {
    pub fn new(max_flow: f64) -> Self {
        Self { position: 0.0, max_flow }
    }

    pub fn set(&mut self, pos: f64) {
        self.position = pos.clamp(0.0, 1.0);
    }

    pub fn flow_rate(&self) -> f64 {
        self.position * self.max_flow
    }
}

#[derive(Clone)]
pub struct Pipe {
    pub pressure: f64,
    pub volume: f64,
    pub inflow: f64,
}

impl Pipe {
    pub fn new(volume: f64) -> Self {
        Self { pressure: 0.0, volume, inflow: 0.0 }
    }

    pub fn step(&mut self, dt: f64, valve: &Valve) {
        let outflow = valve.flow_rate();
        let net = self.inflow - outflow;
        self.pressure += net * dt / self.volume;
        self.pressure = self.pressure.max(0.0);
    }
}

#[derive(Clone)]
pub struct Heater {
    pub power: f64,
    pub temperature: f64,
    pub mass: f64,
    pub specific_heat: f64,
}

impl Heater {
    pub fn new(mass: f64) -> Self {
        Self { power: 0.0, temperature: 20.0, mass, specific_heat: 4184.0 }
    }

    pub fn step(&mut self, dt: f64, coolant_flow: f64, coolant_temp: f64) {
        let energy = self.power * dt;
        let temp_rise = energy / (self.mass * self.specific_heat);
        let cooling = coolant_flow * (self.temperature - coolant_temp) * dt / self.mass;
        self.temperature += temp_rise - cooling;
    }
}

#[derive(Clone, Debug)]
pub enum Phase {
    Liquid,
    Boiling { vapor_fraction: f64 },
    Gas,
}

#[derive(Clone)]
pub struct Boiler {
    pub water_temp: f64,
    pub pressure: f64,
    pub water_mass: f64,
    pub vapor_mass: f64,
    pub heat_input: f64,
}

impl Boiler {
    pub fn new(water_mass: f64) -> Self {
        Self { water_temp: 20.0, pressure: 101325.0, water_mass, vapor_mass: 0.0, heat_input: 0.0 }
    }

    pub fn step(&mut self, dt: f64, heat_kw: f64, water_in: f64) {
        self.heat_input = heat_kw * 1000.0 * dt;
        let energy_per_kg = self.heat_input / self.water_mass.max(0.001);
        self.water_temp += energy_per_kg / 4184.0;

        if self.water_temp >= 100.0 && self.water_mass > 0.0 {
            let vaporize = (self.water_temp - 100.0) * self.water_mass * 0.01;
            let vaporized = vaporize.min(self.water_mass);
            self.water_mass -= vaporized;
            self.vapor_mass += vaporized;
            self.water_temp = 100.0;
            self.pressure = 101325.0 + self.vapor_mass * 1000.0;
        }

        self.water_mass += water_in * dt;
    }

    pub fn phase(&self) -> Phase {
        if self.vapor_mass <= 0.0 && self.water_temp < 99.9 {
            Phase::Liquid
        } else if self.vapor_mass > 0.0 && self.water_mass > 0.0 {
            Phase::Boiling { vapor_fraction: self.vapor_mass / (self.water_mass + self.vapor_mass).max(0.001) }
        } else {
            Phase::Gas
        }
    }
}
