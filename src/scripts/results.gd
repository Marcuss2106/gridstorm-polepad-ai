extends Control

@export var annotated_image_rect : TextureRect
@export var pole_id_label        : RichTextLabel
@export var pole_type_label      : RichTextLabel
@export var components_label     : RichTextLabel
@export var encroachment_label   : RichTextLabel
@export var severity_label       : RichTextLabel

const SEVERITY_NAMES := ["None", "Low", "Medium", "High"]

const form_scene = preload("uid://y33a508r54up")


func _ready() -> void:
	var pd : PoleData = GlobalData.pole_data
	if pd == null:
		return

	# Annotated image
	if pd.ai_annotated_image != null:
		annotated_image_rect.texture = ImageTexture.create_from_image(pd.ai_annotated_image)

	# Pole ID
	var id_text : String = pd.poleID if pd.poleID != "" else "(not detected)"
	pole_id_label.text = "[b]Pole ID:[/b]  " + id_text

	# Pole type
	var type_text : String = pd.ai_pole_type if pd.ai_pole_type != "" else "(unknown)"
	pole_type_label.text = "[b]Pole Type:[/b]  " + type_text.capitalize()

	# Components
	var comps : String
	if pd.ai_components.is_empty():
		comps = "(none detected)"
	else:
		comps = ", ".join(pd.ai_components.map(func(c): return c.capitalize()))
	components_label.text = "[b]Components:[/b]  " + comps

	# Encroachment badge
	if pd.ai_encroachment:
		encroachment_label.text = "[color=#e03030][b]⚠ ENCROACHMENT DETECTED[/b][/color]"
	else:
		encroachment_label.text = "[color=#30a030][b]✔ No Encroachment[/b][/color]"

	# Vegetation severity
	var sev_idx  : int    = clampi(pd.vegetation, 0, SEVERITY_NAMES.size() - 1)
	var sev_name : String = SEVERITY_NAMES[sev_idx]
	severity_label.text = "[b]Vegetation:[/b]  " + sev_name


func _on_edit_button_pressed() -> void:
	get_tree().root.add_child(form_scene.instantiate())
	queue_free()
