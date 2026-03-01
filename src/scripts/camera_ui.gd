extends Control

@export var textlabel : RichTextLabel
@export var subviewport : SubViewport
var pic_count = 0
var first_pic
var second_pic
const next_scene = preload("uid://y33a508r54up")

func _ready() -> void:
	textlabel.text = "[center]Please take a picture of just the pole's tag (plate)."

func _on_camera_button_pressed() -> void:
	var img = subviewport.get_texture().get_image()
	pic_count += 1
	if pic_count == 1:
		first_pic = img
		img.save_jpg("res://tempdata/plate.jpg")
		textlabel.text = "[center]Now take a picture of the whole pole."
	elif pic_count == 2:
		second_pic = img
		img.save_jpg("res://tempdata/pole.jpg")
		GlobalData.pole_data = PoleData.create_from_pics([first_pic, second_pic])
		get_tree().root.add_child(next_scene.instantiate())
		queue_free()
