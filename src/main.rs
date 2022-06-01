mod rtree;

use std::error::Error;
use std::f32::consts::TAU;
use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use rtree::RTree;
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use sdl2::render::{BlendMode, Canvas, TextureCreator};
use sdl2::ttf::Font;
use sdl2::video::{FullscreenType, Window, WindowContext};
use sdl2::EventPump;
use vek::{Aabr, Extent2, Vec2};

use crate::rtree::QueryAction;

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
    pause_time: bool,
    enable_rtree: bool,
    render_rtree: bool,
    rtree: RTree<usize, 8>,
    frame_time: f64,
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

    fn fattened_aabr(&self) -> Aabr<f32> {
        let aabr = self.aabr();
        let v = self.velocity * 0.1;

        let displaced = Aabr {
            min: aabr.min + v,
            max: aabr.max + v,
        };

        aabr.union(displaced)
    }
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
        pause_time: false,
        enable_rtree: true,
        render_rtree: true,
        rtree: RTree::new(),
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
                    Keycode::Space if !state.pressing_space => {
                        state.pressing_space = true;
                        state.pause_time = !state.pause_time;
                    }
                    Keycode::W if !state.pressing_w => {
                        state.pressing_w = true;
                    }
                    Keycode::E if !state.pressing_e => {
                        state.pressing_e = true;
                        state.render_rtree = !state.render_rtree;
                    }
                    Keycode::R if !state.pressing_r => {
                        state.pressing_r = true;
                        state.objects.clear();
                        state.rtree.clear();
                    }
                    Keycode::T if !state.pressing_t => {
                        state.pressing_t = true;
                        state.enable_rtree = !state.enable_rtree;
                        state.rtree.clear();
                        if state.enable_rtree {
                            for (obj_idx, obj) in state.objects.iter().enumerate() {
                                state.rtree.insert(obj_idx, obj.fattened_aabr());
                            }
                        } else {
                            state.rtree.clear();
                        }
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

        if state.pressing_w && state.objects.len() < 100_000 {
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

            if state.enable_rtree {
                state.rtree.insert(state.objects.len(), obj.fattened_aabr());
            }

            state.objects.push(obj);
            state.colliding_buf.push(false);
        }

        let delta = target_fps.recip() * !state.pause_time as u8 as f32;

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

        if state.enable_rtree {
            state.rtree.retain(
                |_| true,
                |&mut idx, fattened_aabr| {
                    let obj = &state.objects[idx];
                    if !fattened_aabr.contains_aabr(obj.aabr()) {
                        *fattened_aabr = obj.fattened_aabr();
                    }
                    true
                },
            );
        }

        state.colliding_buf.clear();
        state.colliding_buf.resize(state.objects.len(), false);

        state
            .colliding_buf
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, c)| {
                let obj = &state.objects[idx];
                let aabr = obj.aabr();

                if state.enable_rtree {
                    state.rtree.query(
                        |other_aabr| other_aabr.collides_with_aabr(aabr),
                        |&other_idx, _| {
                            if other_idx != idx
                                && state.objects[other_idx].aabr().collides_with_aabr(aabr)
                            {
                                *c = true;
                                QueryAction::Break
                            } else {
                                QueryAction::Continue
                            }
                        },
                    );
                } else {
                    for (other_idx, other_obj) in state.objects.iter().enumerate() {
                        if other_idx != idx && aabr.collides_with_aabr(other_obj.aabr()) {
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

        let level_colors = [Color::BLUE, Color::GREEN, Color::MAGENTA, Color::RED];

        let depth = state.rtree.depth();

        if state.enable_rtree && state.render_rtree {
            state.rtree.visit(|aabr, level| {
                let color = level_colors[(depth + 1 - level) % level_colors.len()];
                let min = pos_to_pixel(state.window_dims, aabr.min);
                let max = pos_to_pixel(state.window_dims, aabr.max);

                let _ = state.canvas.set_draw_color(color);
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

        let (rtree_text_color, rtree_text) = if state.enable_rtree {
            (Color::BLACK, "R-Tree: Enabled")
        } else {
            (Color::RED, "R-Tree: Disabled")
        };

        render_text(&mut state, &font, rtree_text_color, rtree_text, (5, 40))?;

        state.canvas.present();

        let elapsed = Instant::now().duration_since(start);

        state.frame_time = elapsed.as_secs_f64();

        thread::sleep(Duration::from_secs_f32(target_fps.recip()).saturating_sub(elapsed));
    }
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
