import dearpygui.dearpygui as dpg
import numpy as np
import os
import copy
import psutil
import torch
from tqdm import tqdm
import time
import json
import cv2
from torchvision import transforms
import threading
import time

from scene.cameras import Camera
from gaussian_renderer import render, render_draw_mouse_click
from scene.draw_gaussians import ObjectModel as DrawGaussians
from scene.base_gaussian import TraceGaussian

to_tensor = transforms.ToTensor()  # auto converts HWC uint8 → CHW float32 in [0,1]

def process_Gaussians(pc):
    means3D = pc.get_xyz
    colors = pc.get_features
    
    opacity = pc.get_opacity

    scales = pc.get_scaling #pc.get_scaling_with_3D_filter
    
    rotations = pc.rotation_activation(pc.splats["quats"])
    
    return means3D, rotations, opacity, colors, scales



class GUIBase:
    """This method servers to intialize the DPG visualization (keeping my code cleeeean!)
    
        Notes:
            none yet...
    """
    def __init__(self, scene,name):
        
        self.gui = True
        self.scene = scene
        self.gaussians = scene.gaussians
        
        self.editor={
            "pencil":{
                "view_flag": False,
                "pc":DrawGaussians()
            },
            "loop":{
                "view_flag": False,
                "pc":TraceGaussian([0., 1., 0.])
            }
        }
        
        self.runname = name
        
        # ---- Layout dimensions ----
        # Main render canvas: reduced width (980x720)
        self.W, self.H = 980, 720
        # Toolbar height beneath the main canvas (this one is draggable)
        self.TOOLBAR_H = 60
        # Total window height
        self.TOTAL_H = self.H + self.TOOLBAR_H
        # Two secondary render canvases stacked vertically, each 512 x (TOTAL_H/2)
        self.W2 = 512
        self.H2 = self.TOTAL_H // 2  # each secondary canvas height
        # Control panel width
        self.CTRL_W = 400
        # Width of the shared right-hand toolbar for the two secondary canvases
        self.RIGHT_TOOLBAR_W = 80
        
        # Initialize the image buffers
        # NOTE: dpg.add_raw_texture expects (H, W, C) layout
        self.buffer_image = np.ones((self.H, self.W, 3), dtype=np.float32)
        self.buffer_image_2 = np.ones((self.H2, self.W2, 3), dtype=np.float32)
        self.buffer_image_3 = np.ones((self.H2, self.W2, 3), dtype=np.float32)
        
        # Other important visualization parameters
        self.vis_mode = 'render'
                
        # Rendering/Novel View Settings
        self.novel_view_background_dir = ""
        self.drag_func = "viewing"
        self.drag_im_buffer = None
        
        # Analysis/Inspection tools
        self.mous_loc = [0, 0] # x,y
        self.mous_loc_last = [0, 0] # x,y
        
        self.design_state = 'add_points' #'viewing'
        self.scale_adjust = 0.07
        
        self.loop_height=1.
        self.loop_radius=1.

        # Viewer settings for camera/view selection
        self.save_frame=False

        # Derive intrinsics from a fixed vertical FOV so they stay correct
        # if the canvas resolution changes. ~32.36° matches the original
        # 1080p calibration (fy=1866.66, cy=540).
        _vfov_deg = 32.36
        _fy = 0.5 * self.H / np.tan(0.5 * np.deg2rad(_vfov_deg))
        _fx = _fy  # square pixels
        _cx = self.W / 2
        _cy = self.H / 2

        self.camera = Camera(
            R=[[
                    -2.821299744937278e-07,
                    0.9659259915351868,
                    0.25881895422935486,
                ],
                [
                    1.0,
                    3.051058001801721e-07,
                    -4.8603780555822595e-08,
                ],
                [
                    -1.2591478082413232e-07,
                    0.25881895422935486,
                    -0.9659259915351868,
                ]], 
            T=[[0.,0.,0.]],
            fx=_fx, fy=_fy,
            cx=_cx, cy=_cy,
            
            width=self.W, height=self.H,

            uid=0,
            data_device=torch.device("cuda"),
            
        )
        
        
        if self.gui:
            print('DPG loading ...')
            dpg.create_context()
            self.register_dpg()
            

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def track_cpu_gpu_usage(self, time):
        # Print GPU and CPU memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 ** 2)  # Convert to MB

        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert to MB
        print(
            f'[{self.stage} {self.iteration}] Time: {time:.2f} | Allocated Memory: {allocated:.2f} MB, Reserved Memory: {reserved:.2f} MB | CPU Memory Usage: {memory_mb:.2f} MB')
    
    def render(self):
        cnt = 0
        if self.gui:
            while dpg.is_dearpygui_running():
                with torch.no_grad():
                    self.viewer_step()
                    dpg.render_dearpygui_frame()    

                    
                with torch.no_grad():
                    self.timer.pause() # log and save
                    torch.cuda.synchronize()
                    if  1000 == 500: # make it 500 so that we dont run this while loading view-test
                        self.track_cpu_gpu_usage(0.1)
                    self.timer.start()
                    
            dpg.destroy_context()
           
    @torch.no_grad()
    def viewer_step(self):
        t0 = time.time()
        mous_hover_value = [0.]

        cam = self.camera # Need to define CAMERA
        
        
        buffer_image = render(
                cam,
                self.gaussians,
                self.editor,
                self.scale_adjust,
                
                view_args={
                    "vis_mode":self.vis_mode,
                    "loop_height":self.loop_height,
                    "loop_radius":self.loop_radius
                        
                },
        )

        

        # Display value of image at current mouse position
        try:
            mous_hover_value = buffer_image[:, self.mous_loc[1], self.mous_loc[0]]
        except:
            mous_hover_value = [0.]
        
        buffer_image = torch.nn.functional.interpolate(
            buffer_image.unsqueeze(0),
            size=(self.H,self.W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    
        
        self.buffer_image = (
            buffer_image.permute(1, 2, 0)
            .contiguous()
            .clamp(0.01, 1)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
        )

        t1 = time.time()
        buffer_image = self.buffer_image

        if self.save_frame:
            frame_uint8 = (buffer_image * 255).astype("uint8")
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite("current_frame.png", frame_bgr)
            self.save_frame = False

        dpg.set_value(
            "_texture", buffer_image
        )  # buffer must be contiguous, else seg fault!
        
        # Update secondary textures (currently just placeholder buffers;
        # replace these when you wire up real second/third sources)
        dpg.set_value("_texture_2", self.buffer_image_2)
        dpg.set_value("_texture_3", self.buffer_image_3)
        
        dpg.set_value("_log_mouse_value", f"({[f'{v:.4f}' for v in mous_hover_value]})")

        # Add _log_view_camera
        if 1./(t1-t0) < 500:
            dpg.set_value("_log_infer_time", f"{1./(t1-t0)} ")

        
        
    def on_image_click(self, button: int, x: int, y: int):
        """Override in subclass to handle clicks on the rendered image.

        button: 0=left, 1=right, 2=middle
        x, y: pixel coordinates in the image
        """
        
        if self.design_state == 'add_points':
            mean, scale, quat = render_draw_mouse_click(
                    self.camera ,
                    self.gaussians,
                    x,y
            )
            if mean.sum().abs() > 0.0001:
                scale = scale
                mean = mean
                self.editor["pencil"]["pc"].add_gaussian(mean, scale, quat, self.gaussians)
    

    
    def register_dpg(self):
        ### register textures
        with dpg.texture_registry(show=False):
            # Main 720p texture
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )
            # Top secondary texture (512 x H/2)
            dpg.add_raw_texture(
                self.W2,
                self.H2,
                self.buffer_image_2,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture_2",
            )
            # Bottom secondary texture (512 x H/2)
            dpg.add_raw_texture(
                self.W2,
                self.H2,
                self.buffer_image_3,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture_3",
            )

        # ---- Layout positions ----
        # Main canvas at (0, 0), size W x H
        # Main toolbar below it at (0, H), size W x TOOLBAR_H  [DRAGGABLE]
        # Control window to the right of main canvas at (W, 0), size CTRL_W x (H + TOOLBAR_H)
        # Top secondary canvas at (W + CTRL_W, 0), size W2 x H2 (where H2 = H/2)
        # Bottom secondary canvas at (W + CTRL_W, H2), size W2 x H2
        # Shared right-hand toolbar at (W + CTRL_W + W2, 0), size RIGHT_TOOLBAR_W x H

        TOTAL_W = self.W + self.CTRL_W + self.W2 + self.RIGHT_TOOLBAR_W
        TOTAL_H = self.TOTAL_H

        ### register window
        # the main rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
            no_resize=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # button theme (shared)
        with dpg.theme() as theme_button:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

        # ---- Toolbar beneath the main canvas (DRAGGABLE) ----
        # This window has no `no_move=True` and shows a title bar so the user
        # can grab it and drag it around. It can also be collapsed.
        with dpg.window(
            label="Main Toolbar",
            tag="_main_toolbar_window",
            width=self.W,
            height=self.TOOLBAR_H,
            pos=[0, self.H],
            no_scrollbar=True,
            no_resize=True,
        ):
            def callback_main_tool_a(sender):
                # TODO: implement
                pass
            def callback_main_tool_b(sender):
                # TODO: implement
                pass
            def callback_main_tool_c(sender):
                # TODO: implement
                pass
            def callback_main_tool_d(sender):
                # TODO: implement
                pass

            with dpg.group(horizontal=True):
                dpg.add_text(" Main : ")
                dpg.add_button(label="Tool A", callback=callback_main_tool_a)
                dpg.add_button(label="Tool B", callback=callback_main_tool_b)
                dpg.add_button(label="Tool C", callback=callback_main_tool_c)
                dpg.add_button(label="Tool D", callback=callback_main_tool_d)

        # ---- Control window (right of main canvas) ----
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=self.CTRL_W,
            height=self.H + self.TOOLBAR_H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
            no_resize=True,
        ):
            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("N/A", tag="_log_infer_time")
            with dpg.group(horizontal=True):
                dpg.add_text("Stage: ")
                dpg.add_text("N/A", tag="_log_stage")

            with dpg.group():
                dpg.add_text("Mode : viewing")

            # ----------------
            #  Control Functions
            # ----------------
            with dpg.collapsing_header(label="Viewer Config", default_open=True):
                
                def callback_toggle_show_pencil(sender):
                    self.editor["pencil"]["view_flag"] = ~self.editor["pencil"]["view_flag"]
                def callback_toggle_show_loop(sender):
                    self.editor["loop"]["view_flag"] = ~self.editor["loop"]["view_flag"]
                    
                def callback_toggle_show_off_editor(sender):
                    for key in self.editor.keys(): self.editor[key]["view_flag"] = False
                    
                    
                dpg.add_text(" : Editor : ")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Off", callback=callback_toggle_show_off_editor)
                    dpg.add_button(label="Penicl", callback=callback_toggle_show_pencil)
                    dpg.add_button(label="Loop", callback=callback_toggle_show_loop)
                     
                    
                def callback_toggle_reset_cam(sender):
                    # TODO: reset camera position
                    pass
                
                def callback_toggle_save_frame(sender):
                        self.save_frame = True
                dpg.add_text(": Frame Settings : ")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="save", callback=callback_toggle_save_frame)
                    dpg.add_button(label="reset", callback=callback_toggle_reset_cam)

                def callback_toggle_show_rgb(sender):
                    self.vis_mode = 'render'
                def callback_toggle_show_depth(sender):
                    self.vis_mode = 'D'
                def callback_toggle_show_XYZ(sender):
                    self.vis_mode = 'xyz'

                dpg.add_text(" : Geometry Buffers : ")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="RGB", callback=callback_toggle_show_rgb)
                    dpg.add_button(label="Zc", callback=callback_toggle_show_depth)
                    dpg.add_button(label="XYZ", callback=callback_toggle_show_XYZ)

            with dpg.collapsing_header(label="Drawing Config", default_open=True):
                dpg.add_text(": Mouse Click : ")

                def callback_toggle_add_point(sender, app_data):
                    if self.design_state != 'add_points':
                        self.design_state = 'add_points'
                    else:
                        self.design_state = 'viewing'
                        
                def callback_toggle_reset_draw_point(sender, app_data):
                    self.editor["pencil"]["pc"].reset()
                    
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Add", callback=callback_toggle_add_point)
                    dpg.add_button(label="Reset", callback=callback_toggle_reset_draw_point)
                
                def callback_scale_adjust(sender, app_data):
                    self.scale_adjust = float(app_data)

                dpg.add_text(": Scale Adjust : ")
                dpg.add_slider_float(
                    label="scale adjust",
                    tag="_slider_scale_adjust",
                    default_value=self.scale_adjust,
                    min_value=0.0,
                    max_value=0.1,
                    callback=callback_scale_adjust,
                )
                
                def callback_lh_adjust(sender, app_data):
                    self.loop_height = float(app_data)
                dpg.add_text(": Loop Height : ")
                dpg.add_slider_float(
                    label="lh adjust",
                    tag="_slider_lh_adjust",
                    default_value=self.loop_height,
                    min_value=self.gaussians.get_xyz[:,2].min().item(),
                    max_value=self.gaussians.get_xyz[:,2].max().item(),
                    callback=callback_lh_adjust,
                )
                def callback_lr_adjust(sender, app_data):
                    self.loop_radius = float(app_data)
                dpg.add_text(": Loop Radius : ")
                dpg.add_slider_float(
                    label="lr adjust",
                    tag="_slider_lr_adjust",
                    default_value=self.loop_radius,
                    min_value=0.0,
                    max_value=4.,
                    callback=callback_lr_adjust,
                )

            # Mouse data display moved into the control window definition
            # so it uses the (now correct) control width.
            with dpg.group(horizontal=True):
                dpg.add_text(" : Mouse data : ")
            with dpg.group(horizontal=True):
                dpg.add_text("Position : ")
                dpg.add_text("N/A", tag="_log_mouse_xy")
            with dpg.group(horizontal=True):
                dpg.add_text("Pixel Value : ")
                dpg.add_text("N/A", tag="_log_mouse_value")

        # ---- Top secondary render canvas (right of control window) ----
        with dpg.window(
            tag="_secondary_window",
            width=self.W2,
            height=self.H2,
            pos=[self.W + self.CTRL_W, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
            no_resize=True,
        ):
            dpg.add_image("_texture_2")

        # ---- Bottom secondary render canvas ----
        with dpg.window(
            tag="_secondary_window_2",
            width=self.W2,
            height=self.H2,
            pos=[self.W + self.CTRL_W, self.H2],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
            no_resize=True,
        ):
            dpg.add_image("_texture_3")

        # ---- Shared right-hand toolbar (vertical, spans both secondary canvases) ----
        with dpg.window(
            tag="_secondary_toolbar_window",
            width=self.RIGHT_TOOLBAR_W,
            height=self.TOTAL_H,
            pos=[self.W + self.CTRL_W + self.W2, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
            no_resize=True,
        ):
            def callback_sec_tool_a(sender):
                # TODO: implement
                pass
            def callback_sec_tool_b(sender):
                # TODO: implement
                pass
            def callback_sec_tool_c(sender):
                # TODO: implement
                pass
            def callback_sec_tool_d(sender):
                # TODO: implement
                pass

            # Stacked vertically so they sit nicely in the narrow right column
            dpg.add_text(" Aux ")
            dpg.add_button(label="Load",  callback=callback_sec_tool_a, width=self.RIGHT_TOOLBAR_W - 16)
            dpg.add_button(label="Clear", callback=callback_sec_tool_b, width=self.RIGHT_TOOLBAR_W - 16)
            dpg.add_button(label="Save",  callback=callback_sec_tool_c, width=self.RIGHT_TOOLBAR_W - 16)
            dpg.add_button(label="Swap",  callback=callback_sec_tool_d, width=self.RIGHT_TOOLBAR_W - 16)
        # ---- Mouse / keyboard handlers (unchanged behavior, only main canvas) ----
        def drag_callback(sender, app_data):
            
            if dpg.is_item_hovered("_primary_window"):

                if self.drag_im_buffer is not None:
                    mouse_hover_value = self.drag_im_buffer[:3, self.mous_loc[1], self.mous_loc[0]].sum()
                else:
                    mouse_hover_value = 0.0

                view_drag_thresh = 0.5
                if mouse_hover_value < view_drag_thresh:
                    button, rel_x, rel_y = app_data
                    cam = self.camera

                    yaw_speed   = 0.001
                    pitch_speed = 0.001
                    cam.yaw   -= rel_x * yaw_speed
                    cam.pitch += rel_y * pitch_speed
                    cam.pitch  = np.clip(cam.pitch, -np.pi/2 + 0.01, np.pi/2 - 0.01)  # never hit poles

                    # Camera position on sphere around world origin (Z-up)
                    cam.T = np.array([
                        cam.orbit_radius * np.sin(cam.yaw) * np.cos(cam.pitch),
                        cam.orbit_radius * np.cos(cam.yaw) * np.cos(cam.pitch),
                        cam.orbit_radius * np.sin(cam.pitch),
                    ], dtype=np.float32)

                    # Look-at with fixed Z-up — no roll ever
                    forward = -cam.T / np.linalg.norm(cam.T)
                    world_up = np.array([0, 0, -1], dtype=np.float32)
                    right = np.cross(forward, world_up)
                    right /= np.linalg.norm(right)
                    up = np.cross(right, forward)
                    up /= np.linalg.norm(up)

                    cam.R = np.stack([right, up, forward], axis=1).astype(np.float32)      
            
        
        def zoom_callback_fov(sender, app_data):
            delta = app_data  # scroll: +1 = up (zoom in), -1 = down (zoom out)

            if dpg.is_item_hovered("_primary_window"):
                if delta > 0:
                    self.camera.orbit_radius += 1
                elif delta < 0:
                    self.camera.orbit_radius = self.camera.orbit_radius - 1 if self.camera.orbit_radius > 1 else 1

                drag_callback(None, (1, 0., 0.))
        
        
        def mouse_hover_callback(sender, app_data):
            # app_data: (x, y) coordinates of mouse position in global viewport
            x, y = app_data

            if dpg.is_item_hovered("_primary_window"):
                self.mous_loc_last = self.mous_loc
                self.mous_loc = [int(x),int(y)]
                dpg.set_value("_log_mouse_xy", f"({x:.1f}, {y:.1f})")

        def mouse_click_callback(sender, app_data):
            # app_data: mouse button index (0=left, 1=right, 2=middle)
            if dpg.is_item_hovered("_primary_window") and self.editor["pencil"]["view_flag"]:
                x, y = self.mous_loc
                button = app_data
                self.on_image_click(button, x, y)

        def key_press_callback(sender, app_data):
            # app_data is the key code
            if app_data == dpg.mvKey_Return and self.editor["loop"]["view_flag"]:
                self.editor["loop"]["pc"].set_loop(self.gaussians)
                
        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=zoom_callback_fov)
            dpg.add_mouse_drag_handler(callback=drag_callback)
            dpg.add_mouse_move_handler(callback=mouse_hover_callback)
            dpg.add_mouse_click_handler(callback=mouse_click_callback)
            dpg.add_key_press_handler(callback=key_press_callback)            
        
        dpg.create_viewport(
            title=f"{self.runname}",
            width=TOTAL_W,
            height=TOTAL_H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme — only applied to the canvas windows so the
        ### control + toolbars keep normal padding for buttons/sliders.
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)
        dpg.bind_item_theme("_secondary_window", theme_no_padding)
        dpg.bind_item_theme("_secondary_window_2", theme_no_padding)

        dpg.setup_dearpygui()

        dpg.show_viewport()

        
        
from scipy.ndimage import distance_transform_edt
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat

from scipy.ndimage import distance_transform_edt
@torch.no_grad()
def get_in_view_dyn_mask(camera, xyz, X, Y) -> torch.Tensor:
    device = xyz.device
    N = xyz.shape[0]

    # Convert to homogeneous coordinates
    xyz_h = torch.cat([xyz, torch.ones((N, 1), device=device)], dim=-1)  # (N, 4)

    # World → Camera (OpenCV convention: +Z forward)
    c2w = camera.pose
    w2c = get_viewmat(c2w[None])[0]
    xyz_cam = (xyz_h @ w2c.T)[:, :3]

    # Only keep points in front of the camera
    in_front = xyz_cam[:, 2] > 0

    # Camera → Pixel (using intrinsics)
    K = torch.from_numpy(camera.K).to(device=device, dtype=torch.float32)
    xy = xyz_cam @ K.T  # (N, 3)
    px = (xy[:, 0] / xy[:, 2]).long()
    py = (xy[:, 1] / xy[:, 2]).long()

    # Visibility check (inside image bounds)
    in_bounds = (
        (px >= 0) & (px < camera.image_width) &
        (py >= 0) & (py < camera.image_height)
    )
    visible_mask = in_front & in_bounds

    # Valid pixel indices
    valid_idx = visible_mask.nonzero(as_tuple=True)[0]
    if len(valid_idx) == 0:
        print("No visible points found.")
        return torch.zeros((camera.image_height, camera.image_width, 3), device=device)

    px_valid = px[valid_idx]
    py_valid = py[valid_idx]

    # Scene occlusion mask (optional)
    mask = (1. - camera.sceneoccluded_mask).to(device).squeeze(0)

    sampled_mask = mask[py_valid, px_valid] > 0.5

    # Projected XYZ image
    H, W = camera.image_height, camera.image_width
    xyz_img = torch.zeros((H, W, 3), device=device)

    px_final = px_valid[sampled_mask]
    py_final = py_valid[sampled_mask]
    xyz_vals = xyz[valid_idx][sampled_mask]
    xyz_img[py_final, px_final] = xyz_vals

    # Visualization (optional)
    show = False
    if show:
        import matplotlib.pyplot as plt
        img_np = xyz_img.detach().cpu().numpy()

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(img_np)
        ax[0].set_title("Projected XYZ (OpenCV)")
        ax[0].axis("off")

        ax[1].imshow(img_np)
        ax[1].scatter(X, Y, s=3, c="red")
        ax[1].set_title("With XY indexing")
        ax[1].axis("off")

        plt.show()
        exit()
        return None
    
    # --- Nearest-neighbor fill for empty pixels ---
    xyz_np = xyz_img.cpu().numpy()   # [H, W, 3]
    valid_mask = (xyz_np.sum(axis=-1) != 0)

    # distance_transform_edt returns for each empty pixel the index of the nearest valid pixel
    dist, indices = distance_transform_edt(~valid_mask,
                                           return_indices=True)
    filled = xyz_np[indices[0], indices[1]]  # nearest xyz per pixel

    point = filled[Y, X, :]
    
    show = False
    if show:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Scatter all visible point cloud
        ax.scatter(
            xyz_vals[:, 0].cpu(),
            xyz_vals[:, 1].cpu(),
            xyz_vals[:, 2].cpu(),
            s=1, c="blue", alpha=0.5, label="Point cloud"
        )

        # Scatter your selected points
        ax.scatter(
            point[ 0],
            point[ 1],
            point[ 2],
            s=60, c="red", marker="o", label="Filtered XYZ"
        )

        ax.set_title("3D Point Cloud with Filtered Points")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        plt.show()
        exit()

    return torch.from_numpy(point).float().to(device)


def remove_screen_points(camera, xyz):
    device = xyz.device
    N = xyz.shape[0]

    # Convert to homogeneous coordinates
    xyz_h = torch.cat([xyz, torch.ones((N, 1), device=device)], dim=-1)  # (N, 4)

    # Apply full projection (world → clip space)
    proj_xyz = xyz_h @ camera.full_proj_transform.to(device)  # (N, 4)

    # Homogeneous divide to get NDC coordinates
    ndc = proj_xyz[:, :3] / proj_xyz[:, 3:4]  # (N, 3)

    in_front = proj_xyz[:, 2] > 0
    in_ndc_bounds = (
        (ndc[:, 0].abs() <= 1) &
        (ndc[:, 1].abs() <= 1) &
        (ndc[:, 2].abs() <= 1)
    )
    visible_mask = in_front & in_ndc_bounds

    # Pixel coordinates for all points (will clamp to bounds)
    px = (((ndc[:, 0] + 1) / 2) * camera.image_width).long().clamp(0, camera.image_width - 1)
    py = (((ndc[:, 1] + 1) / 2) * camera.image_height).long().clamp(0, camera.image_height - 1)

    # Scene mask (1 = free, 0 = masked/occluded)
    mask_img = (camera.sceneoccluded_mask).to(device).squeeze(0)

    # Start with all points marked False (not removed)
    remove_mask = torch.zeros(N, dtype=torch.bool, device=device)

    # Only check points that are visible
    sampled_mask = mask_img[py[visible_mask], px[visible_mask]].bool()

    # Mark visible points inside the mask for removal
    remove_mask[visible_mask] = sampled_mask

    return remove_mask