extends Control

# ── Set this to your dev machine's local IP before building ──────────────────
const ApiConfig   := preload("res://scripts/api_config.gd")
const API_HOST    := ApiConfig.API_HOST
const API_PORT    := ApiConfig.API_PORT
var   ANALYZE_URL := API_HOST + ":" + str(API_PORT) + "/analyze"

@export var textlabel  : RichTextLabel
@export var subviewport: SubViewport
@export var camera_node : CanvasItem

var pic_count  := 0
var first_pic  : Image
var second_pic : Image
var _http      : HTTPRequest


func _ready() -> void:
	textlabel.text = "[center]Please take a picture of just the pole's tag (plate)."
	_http = HTTPRequest.new()
	add_child(_http)
	_http.request_completed.connect(_on_request_completed)
	camera_node.visible = true


func _on_camera_button_pressed() -> void:
	var img = subviewport.get_texture().get_image()
	pic_count += 1

	if pic_count == 1:
		first_pic = img
		img.save_jpg("user://tempdata/plate.jpg")
		textlabel.text = "[center]Now take a picture of the whole pole."

	elif pic_count == 2:
		second_pic = img
		img.save_jpg("user://tempdata/pole.jpg")
		GlobalData.pole_data = PoleData.create_from_pics([first_pic, second_pic])
		camera_node.visible = false
		_send_to_api(first_pic, second_pic)


func _send_to_api(plate_img: Image, pole_img: Image) -> void:
	textlabel.text = "[center]Analyzing... please wait."
	_set_button_enabled(false)

	# Encode both images as JPEG bytes then base64
	var plate_b64 : String = Marshalls.raw_to_base64(plate_img.save_jpg_to_buffer(0.9))
	var pole_b64  : String = Marshalls.raw_to_base64(pole_img.save_jpg_to_buffer(0.9))

	var body    : String            = JSON.stringify({"plate_image_b64": plate_b64, "pole_image_b64": pole_b64})
	var headers : PackedStringArray = ["Content-Type: application/json"]

	var err : int = _http.request(ANALYZE_URL, headers, HTTPClient.METHOD_POST, body)
	if err != OK:
		_show_error("Could not connect to AI server (error %d).\nCheck that the server is running and API_HOST is correct." % err)


func _on_request_completed(
	result       : int,
	response_code: int,
	_headers     : PackedStringArray,
	body         : PackedByteArray,
) -> void:
	if result != HTTPRequest.RESULT_SUCCESS or response_code != 200:
		_show_error("Server returned an error (HTTP %d). Check the server logs." % response_code)
		return

	var data : Variant = JSON.parse_string(body.get_string_from_utf8())
	if data == null or not data is Dictionary:
		_show_error("Could not parse server response.")
		return

	var pd : PoleData = GlobalData.pole_data

	# Populate standard form fields from AI results so the form is pre-filled
	pd.poleID      = data.get("pole_id",  "")
	pd.poleType    = data.get("pole_type", "")
	pd.vegetation  = data.get("vegetation_severity", 0)
	pd.surroundings = []
	for comp in data.get("detected_components", []):
		pd.surroundings.append(str(comp))

	# AI-specific fields
	pd.ai_pole_type    = data.get("pole_type", "")
	pd.ai_encroachment = data.get("encroachment", false)
	pd.ai_components   = []
	for comp in data.get("detected_components", []):
		pd.ai_components.append(str(comp))

	# Per-field confidence scores (keyed by field name, values 0.0 – 1.0)
	var raw_conf = data.get("confidences", {})
	if raw_conf is Dictionary:
		pd.confidences = raw_conf

	# Decode annotated image
	var b64 : String = data.get("annotated_image_b64", "")
	if b64 != "":
		var jpeg_bytes : PackedByteArray = Marshalls.base64_to_raw(b64)
		var annotated  : Image           = Image.new()
		if annotated.load_jpg_from_buffer(jpeg_bytes) == OK:
			pd.ai_annotated_image = annotated

	# Navigate to the results screen
	var results_scene = load("res://screens/results.tscn")
	get_tree().root.add_child(results_scene.instantiate())
	queue_free()


func _show_error(msg: String) -> void:
	textlabel.text = "[center][color=red]" + msg + "[/color]\n\nTap to retry."
	_set_button_enabled(true)
	pic_count = 1  # let the user retake the second photo


func _set_button_enabled(enabled: bool) -> void:
	var btn : Button = get_node_or_null("CameraButton")
	if btn:
		btn.disabled = not enabled
