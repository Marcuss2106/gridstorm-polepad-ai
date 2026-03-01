extends Button

@export var to_scene : PackedScene
@export var this_root : Node

func _pressed() -> void:
	get_tree().root.add_child(to_scene.instantiate())
