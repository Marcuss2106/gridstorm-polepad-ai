extends Node

var pole_data : PoleData

func _ready() -> void:
	var dir = DirAccess.open("user://")
	if !dir.dir_exists("data"):
		dir.make_dir("data")
	if !dir.dir_exists("tempdata"):
		dir.make_dir("tempdata")
