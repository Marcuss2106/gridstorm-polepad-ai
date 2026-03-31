extends Button

@export var this_root : Node
const MAIN_MENU = preload("res://screens/main.tscn")

func _pressed() -> void:
	if GlobalData.pole_data != null:
		if FileAccess.file_exists("user://data/pole_"+GlobalData.pole_data.poleID+".res"):
			var dir = DirAccess.open("user://data/")
			dir.remove("pole_"+GlobalData.pole_data.poleID+".res")
	get_tree().root.add_child(MAIN_MENU.instantiate())
	this_root.queue_free()
