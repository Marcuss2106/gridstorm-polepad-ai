extends Button

@export var this_root : Node
const MAIN_MENU = preload("res://screens/main.tscn")

func _pressed() -> void:
	get_tree().root.add_child(MAIN_MENU.instantiate())
	this_root.queue_free()
