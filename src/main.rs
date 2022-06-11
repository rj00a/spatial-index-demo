mod bvh_basic;
mod rtree;

use std::error::Error;
use std::f32::consts::TAU;
use std::thread;
use std::time::{Duration, Instant};

use bvh_basic::Bvh;
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use sdl2::render::{BlendMode, Canvas, TextureCreator};
use sdl2::ttf::Font;
use sdl2::video::{FullscreenType, Window, WindowContext};
use sdl2::EventPump;
use vek::{Aabr, Extent2, Vec2};

struct State {
    canvas: Canvas<Window>,
    texture_creator: TextureCreator<WindowContext>,
    event_pump: EventPump,
    window_dims: Extent2<u32>,
    pressing_space: bool,
    pressing_w: bool,
    pressing_e: bool,
    pressing_r: bool,
    pressing_t: bool,
    pressing_y: bool,
    pause_time: bool,
    render_bvh: bool,
    render_objects: bool,
    bvh: Bvh<usize>,
    target_fps: f64,
    displayed_fps: f64,
    last_fps_update: Instant,
    fps_sample_sum: f64,
    fps_sample_count: usize,
    objects: Vec<Object>,
    colliding_buf: Vec<bool>,
}

struct Object {
    pos: Vec2<f32>,
    dims: Extent2<f32>,
    velocity: Vec2<f32>,
}

impl Object {
    fn aabr(&self) -> Aabr<f32> {
        Aabr {
            min: self.pos - self.dims / 2.0,
            max: self.pos + self.dims / 2.0,
        }
    }

    // fn fattened_aabr(&self) -> Aabr<f32> {
    //     let aabr = self.aabr();
    //     let v = self.velocity * 0.1;

    //     let displaced = Aabr {
    //         min: aabr.min + v,
    //         max: aabr.max + v,
    //     };

    //     aabr.union(displaced)
    // }
}

fn main() -> Result<(), Box<dyn Error>> {
    let sdl_context = sdl2::init().unwrap();
    let sdl_ttf = sdl2::ttf::init()?;

    let font = sdl_ttf.load_font("UbuntuMono-Regular.ttf", 20)?;

    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem
        .window("R-Tree Demo", 1280, 720)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().accelerated().build().unwrap();
    canvas.set_blend_mode(BlendMode::Blend);

    let event_pump = sdl_context.event_pump().unwrap();

    let window_dims = canvas.window().size().into();

    let texture_creator = canvas.texture_creator();

    let mut state = State {
        canvas,
        texture_creator,
        event_pump,
        window_dims,
        pressing_space: false,
        pressing_w: false,
        pressing_e: false,
        pressing_r: false,
        pressing_t: false,
        pressing_y: false,
        pause_time: false,
        render_bvh: true,
        render_objects: true,
        bvh: Bvh::new(),
        target_fps: 144.0,
        displayed_fps: 144.0,
        last_fps_update: Instant::now(),
        fps_sample_sum: 0.0,
        fps_sample_count: 0,
        objects: Vec::new(),
        colliding_buf: Vec::new(),
    };

    loop {
        let start = Instant::now();

        for event in state.event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => return Ok(()),
                Event::Window {
                    win_event: WindowEvent::SizeChanged(w, h),
                    ..
                } => {
                    if w > 0 && h > 0 {
                        state.window_dims.w = w as u32;
                        state.window_dims.h = h as u32;
                    }
                }
                Event::KeyDown {
                    keycode: Some(kc), ..
                } => match kc {
                    Keycode::Escape => return Ok(()),
                    Keycode::F11 => match state.canvas.window().fullscreen_state() {
                        FullscreenType::Off => state
                            .canvas
                            .window_mut()
                            .set_fullscreen(FullscreenType::Desktop)?,
                        FullscreenType::True | FullscreenType::Desktop => state
                            .canvas
                            .window_mut()
                            .set_fullscreen(FullscreenType::Off)?,
                    },
                    Keycode::Space if !state.pressing_space => {
                        state.pressing_space = true;
                        state.pause_time = !state.pause_time;
                    }
                    Keycode::W if !state.pressing_w => {
                        state.pressing_w = true;
                    }
                    Keycode::E if !state.pressing_e => {
                        state.pressing_e = true;
                        state.render_bvh = !state.render_bvh;
                    }
                    Keycode::R if !state.pressing_r => {
                        state.pressing_r = true;
                        state.objects.clear();
                    }
                    Keycode::T if !state.pressing_t => {
                        state.pressing_y = true;
                        state.render_objects = !state.render_objects;
                    }
                    _ => {}
                },
                Event::KeyUp {
                    keycode: Some(kc), ..
                } => match kc {
                    Keycode::Space => state.pressing_space = false,
                    Keycode::W => state.pressing_w = false,
                    Keycode::E => state.pressing_e = false,
                    Keycode::R => state.pressing_r = false,
                    Keycode::T => state.pressing_t = false,
                    Keycode::Y => state.pressing_y = false,
                    _ => {}
                },
                _ => {}
            }
        }

        let aspect_ratio = state.window_dims.w as f32 / state.window_dims.h as f32;

        const MAX_RECT_SIZE: f32 = 0.01;

        let arena_bounds = Aabr {
            min: Vec2::new(-aspect_ratio / 2.0, -0.5),
            max: Vec2::new(aspect_ratio / 2.0, 0.5),
        };

        if state.pressing_w && state.objects.len() < 500_000 {
            let mut rng = rand::thread_rng();

            let dims = Extent2::new(
                rng.gen_range(0.003..=MAX_RECT_SIZE),
                rng.gen_range(0.003..=MAX_RECT_SIZE),
            );

            let center = Vec2::new(
                rng.gen_range(
                    arena_bounds.min.x + dims.w / 2.0..=arena_bounds.max.x - dims.w / 2.0,
                ),
                rng.gen_range(
                    arena_bounds.min.y + dims.h / 2.0..=arena_bounds.max.y - dims.h / 2.0,
                ),
            );

            let speed = rng.gen_range(0.01..=0.05);
            let angle = rng.gen_range(0.0..TAU);

            let obj = Object {
                pos: center,
                dims,
                velocity: Vec2::new(angle.cos(), angle.sin()) * speed,
            };

            // if state.enable_rtree {
            //     state.rtree.insert(state.objects.len(), obj.fattened_aabr());
            // }

            state.objects.push(obj);
            state.colliding_buf.push(false);
        }

        let delta = state.target_fps.recip() as f32 * !state.pause_time as u8 as f32;

        for obj in state.objects.iter_mut() {
            obj.pos += obj.velocity * delta;

            if obj.pos.x < arena_bounds.min.x || obj.pos.x > arena_bounds.max.x {
                obj.pos.x = obj.pos.x.clamp(arena_bounds.min.x, arena_bounds.max.x);
                obj.velocity.x *= -1.0;
            }

            if obj.pos.y < arena_bounds.min.y || obj.pos.y > arena_bounds.max.y {
                obj.pos.y = obj.pos.y.clamp(arena_bounds.min.y, arena_bounds.max.y);
                obj.velocity.y *= -1.0;
            }
        }

        let start_rebuild = Instant::now();

        state.bvh.build(
            state
                .objects
                .iter()
                .enumerate()
                .map(|(idx, obj)| (idx, obj.aabr())),
        );

        state.colliding_buf.clear();
        state.colliding_buf.resize(state.objects.len(), false);

        let start_query = Instant::now();

        state
            .colliding_buf
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, c)| {
                let obj = &state.objects[idx];
                let aabr = obj.aabr();

                if state
                    .bvh
                    .find(
                        |other_aabr| other_aabr.collides_with_aabr(aabr),
                        |&other_idx, _| (other_idx != idx).then(|| ()),
                    )
                    .is_some()
                {
                    *c = true;
                }
            });

        let done_query = Instant::now();

        assert_eq!(state.objects.len(), state.colliding_buf.len());

        render(&mut state, &font)?;

        let now = Instant::now();
        let elapsed = now.duration_since(start);

        if now - state.last_fps_update > Duration::from_secs_f64(0.5) {
            state.last_fps_update = now;
            state.displayed_fps = state.fps_sample_sum / state.fps_sample_count as f64;
            state.fps_sample_sum = 0.0;
            state.fps_sample_count = 0;

            let rebuild_duration = start_query - start_rebuild;
            let query_duration = done_query - start_query;

            let rebuild_fraction = rebuild_duration.as_secs_f64() / elapsed.as_secs_f64();
            let query_fraction = query_duration.as_secs_f64() / elapsed.as_secs_f64();

            println!(
                "FPS: {:.2}, rebuild: {:.2}%, query: {:.2}%",
                state.displayed_fps,
                rebuild_fraction * 100.0,
                query_fraction * 100.0,
            );
        }

        state.fps_sample_sum += elapsed.as_secs_f64().recip();
        state.fps_sample_count += 1;

        thread::sleep(Duration::from_secs_f64(state.target_fps.recip()).saturating_sub(elapsed));
    }
}

fn render(state: &mut State, font: &Font) -> Result<(), Box<dyn Error>> {
    state.canvas.set_draw_color(Color::WHITE);
    state.canvas.clear();

    if state.render_objects {
        for (obj, &collides) in state.objects.iter().zip(&state.colliding_buf) {
            let top_left_pixel = pos_to_pixel(
                state.window_dims,
                Vec2::new(obj.pos.x - obj.dims.w / 2.0, obj.pos.y + obj.dims.h / 2.0),
            );

            let bottom_right_pixel = pos_to_pixel(
                state.window_dims,
                Vec2::new(obj.pos.x + obj.dims.w / 2.0, obj.pos.y - obj.dims.h / 2.0),
            );

            let color = if collides {
                Color::RGBA(255, 0, 0, 128)
            } else {
                Color::RGBA(0, 0, 0, 128)
            };

            state.canvas.set_draw_color(color);
            state.canvas.fill_rect(Rect::new(
                top_left_pixel.x,
                top_left_pixel.y,
                bottom_right_pixel.x.abs_diff(top_left_pixel.x),
                bottom_right_pixel.y.abs_diff(top_left_pixel.y),
            ))?;
        }
    }

    let depth_colors = [Color::RED, Color::MAGENTA, Color::GREEN, Color::BLUE];

    if state.render_bvh {
        state.bvh.visit(|aabr, depth| {
            let color = depth_colors[depth % depth_colors.len()];

            let min = pos_to_pixel(state.window_dims, aabr.min);
            let max = pos_to_pixel(state.window_dims, aabr.max);

            state.canvas.set_draw_color(color);
            let _ = state.canvas.draw_rect(Rect::new(
                min.x,
                max.y,
                min.x.abs_diff(max.x),
                min.y.abs_diff(max.y),
            ));
        });
    }

    let object_count = state.objects.len();

    render_text(
        state,
        font,
        Color::BLACK,
        &format!("Objects: {object_count}"),
        (5, 0),
    )?;

    let avg_fps = state.displayed_fps;

    let fps_color = if avg_fps < state.target_fps {
        Color::RGB(217, 20, 20)
    } else {
        Color::BLACK
    };

    render_text(
        state,
        font,
        fps_color,
        &format!("FPS: {:.2}", avg_fps),
        (5, 20),
    )?;

    state.canvas.present();

    Ok(())
}

/// Screen space to world space.
#[allow(unused)]
fn pixel_to_pos(window_dims: Extent2<u32>, pos: Vec2<i32>) -> Vec2<f32> {
    (pos.as_() - window_dims.as_() / 2.0) / window_dims.h as f32 * Vec2::new(1.0, -1.0)
}

/// World space to screen space.
fn pos_to_pixel(window_dims: Extent2<u32>, pos: Vec2<f32>) -> Vec2<i32> {
    (pos * Vec2::new(1.0, -1.0) * window_dims.h as f32 + window_dims.as_() / 2.0)
        .round()
        .as_()
}

fn render_text(
    state: &mut State,
    font: &Font,
    color: Color,
    text: &str,
    top_left: impl Into<Vec2<i32>>,
) -> Result<(), Box<dyn Error>> {
    let texture = state
        .texture_creator
        .create_texture_from_surface(font.render(text).blended(color)?)?;

    let (width, height) = font.size_of(text)?;

    let top_left = top_left.into();
    state.canvas.copy(
        &texture,
        None,
        Rect::new(top_left.x, top_left.y, width, height),
    )?;

    Ok(())
}
