import bpy
import json
import time

class SignVerseBlenderClient:
    """
    Real-time 3D Pose Visualization for Blender (Sign-Verse Client)
    """
    def __init__(self, obj_name="SignVersePoints"):
        self.points_obj = bpy.data.collections.new(obj_name)
        bpy.context.scene.collection.children.link(self.points_obj)
        self.meshes = []

    def clear_scene(self):
        """
        Clears previous pose data.
        """
        for obj in self.points_obj.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

    def render_pose(self, landmarks):
        """
        Renders 3D landmarks as spheres in Blender.
        """
        self.clear_scene()
        for i, lm in enumerate(landmarks):
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=(lm['x'], lm['y'], lm['z']))
            obj = bpy.context.active_object
            obj.name = f"LM_{i}"
            self.points_obj.objects.link(obj)

    def listen_and_visualize(self, data_file="/tmp/sign_verse_data.json"):
        """
        Periodically listens for new 3D data and visualizes it inside Blender.
        """
        print("Starting Sign-Verse Blender visualization...")
        while True:
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    self.render_pose(data['landmarks'])
                    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                time.sleep(0.03) # ~30 FPS
            except (IOError, KeyError):
                time.sleep(0.1)

if __name__ == "__main__":
    client = SignVerseBlenderClient()
    # client.listen_and_visualize()
