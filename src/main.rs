mod rtree;

use std::error::Error;
use std::f32::consts::TAU;
use std::thread;
use std::time::{Duration, Instant};

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
    pressing_r: bool,
    pressing_t: bool,
    enable_rtree: bool,
    frame_time: f64,
    objects: Vec<Object>,
    colliding_buf: Vec<bool>,
}

struct Object {
    pos: Vec2<f32>,
    dims: Extent2<f32>,
    velocity: Vec2<f32>,
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
        pressing_r: false,
        pressing_t: false,
        enable_rtree: true,
        frame_time: 0.0,
        objects: Vec::new(),
        colliding_buf: Vec::new(),
    };

    let target_fps = 144.0_f32;

    loop {
        let start = Instant::now();

        state.canvas.set_draw_color(Color::WHITE);
        state.canvas.clear();

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
                    Keycode::Space => state.pressing_space = true,
                    Keycode::R if !state.pressing_r => {
                        state.pressing_r = true;
                        state.objects.clear();
                    }
                    Keycode::T if !state.pressing_t => {
                        // TODO: clear R-Tree
                        state.pressing_t = true;
                        state.enable_rtree = !state.enable_rtree;
                    }
                    _ => {}
                },
                Event::KeyUp {
                    keycode: Some(kc), ..
                } => match kc {
                    Keycode::Space => state.pressing_space = false,
                    Keycode::R => state.pressing_r = false,
                    Keycode::T => state.pressing_t = false,
                    _ => {}
                },
                _ => {}
            }
        }

        let aspect_ratio = state.window_dims.w as f32 / state.window_dims.h as f32;

        const MAX_RECT_SIZE: f32 = 0.01;

        let arena_bounds = Aabr {
            min: Vec2::new(-aspect_ratio / 2.0, -0.5) + MAX_RECT_SIZE,
            max: Vec2::new(aspect_ratio / 2.0, 0.5) - MAX_RECT_SIZE,
        };

        if state.pressing_space && state.objects.len() < 100_000 {
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

            let speed = rng.gen_range(0.01..=0.1);
            let angle = rng.gen_range(0.0..TAU);

            state.objects.push(Object {
                pos: center,
                dims,
                velocity: Vec2::new(angle.cos(), angle.sin()) * speed,
            });
            state.colliding_buf.push(false);
        }

        let delta = target_fps.recip();

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

        state.colliding_buf.clear();
        state.colliding_buf.resize(state.objects.len(), false);

        state
            .colliding_buf
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, c)| {
                let obj = &state.objects[idx];
                let aabr = Aabr {
                    min: obj.pos - obj.dims / 2.0,
                    max: obj.pos + obj.dims / 2.0,
                };

                for (other_idx, other_obj) in state.objects.iter().enumerate() {
                    if other_idx != idx {
                        let other_aabr = Aabr {
                            min: other_obj.pos - other_obj.dims / 2.0,
                            max: other_obj.pos + other_obj.dims / 2.0,
                        };

                        if aabr.collides_with_aabr(other_aabr) {
                            *c = true;
                            break;
                        }
                    }
                }
            });

        assert_eq!(state.objects.len(), state.colliding_buf.len());

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
                Color::RGBA(0, 0, 255, 128)
            };

            state.canvas.set_draw_color(color);
            state.canvas.fill_rect(Rect::new(
                top_left_pixel.x,
                top_left_pixel.y,
                bottom_right_pixel.x.abs_diff(top_left_pixel.x),
                bottom_right_pixel.y.abs_diff(top_left_pixel.y),
            ))?;
        }

        let object_count = state.objects.len();

        render_text(
            &mut state,
            &font,
            Color::BLACK,
            &format!("Objects: {object_count}"),
            (5, 0),
        )?;

        let frame_time = state.frame_time as f32;

        let frame_time_color = if frame_time > target_fps.recip() {
            Color::RED
        } else {
            Color::BLACK
        };

        render_text(
            &mut state,
            &font,
            frame_time_color,
            &format!("Frame Time: {:.2} FPS", frame_time.recip()),
            (5, 20),
        )?;

        state.canvas.present();

        let elapsed = Instant::now().duration_since(start);

        state.frame_time = elapsed.as_secs_f64();

        thread::sleep(Duration::from_secs_f32(target_fps.recip()).saturating_sub(elapsed));
    }
}

/// Screen space to world space.
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
