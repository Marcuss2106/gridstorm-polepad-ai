extends Control

# ── Set to match camera_ui.gd ────────────────────────────────────────────────
const API_HOST   := "http://10.35.241.238"
const API_PORT   := 8000
var   SUBMIT_URL := API_HOST + ":" + str(API_PORT) + "/submit"

# ── Wired in form.tscn ───────────────────────────────────────────────────────
@export var banner_label : Label
@export var field_vbox   : VBoxContainer
@export var submit_btn   : Button

# ── Internal state ───────────────────────────────────────────────────────────
var _pole_data    : PoleData
var _confirmed    : Dictionary   # field_id -> bool
var _input_nodes  : Dictionary   # field_id -> input widget node
var _conf_bars    : Dictionary   # field_id -> ColorRect (left accent bar)
var _conf_labels  : Dictionary   # field_id -> Label (shows "87%")
var _confirm_btns : Dictionary   # field_id -> Button

var _http         : HTTPRequest
var _total_fields := 0

# Field definitions: [id, display_label, widget_type, confidence_key]
# widget_type: "text" | "checkbox" | "option" | "encr_check"
const FIELDS := [
	["pole_id",           "Pole ID",              "text",       "pole_id"],
	["pole_type",         "Pole Type",             "text",       "pole_type"],
	["transformer",       "Transformer",           "checkbox",   "transformer"],
	["insulator",         "Insulator",             "checkbox",   "insulator"],
	["streetlight",       "Streetlight",           "checkbox",   "streetlight"],
	["vegetation",        "Vegetation Severity",   "option",     "vegetation_severity"],
	["encroachment",      "Encroachment",          "checkbox",   "encroachment"],
]

# ── Confidence color thresholds ───────────────────────────────────────────────
const COLOR_GREEN := Color(0.20, 0.70, 0.30)   # >= 0.85  success
const COLOR_AMBER := Color(0.85, 0.60, 0.10)   # >= 0.60  caution
const COLOR_RED   := Color(0.80, 0.20, 0.20)   # <  0.60  needs attention

func _conf_color(c: float) -> Color:
	if c >= 0.85:
		return COLOR_GREEN
	if c >= 0.60:
		return COLOR_AMBER
	return COLOR_RED


func _ready() -> void:
	_http = HTTPRequest.new()
	add_child(_http)
	_http.request_completed.connect(_on_submit_completed)

	_pole_data = GlobalData.pole_data
	if _pole_data == null:
		_pole_data = PoleData.create_empty_pole()

	_build_photo_row()
	_build_field_rows()
	_update_banner()
	_update_submit_label()

func save_data():
	ResourceSaver.save(_pole_data, "data/pole_"+_pole_data.poleID+".res")


# ── Photo preview row ─────────────────────────────────────────────────────────

func _build_photo_row() -> void:
	var hbox := HBoxContainer.new()
	hbox.add_theme_constant_override("separation", 8)
	hbox.custom_minimum_size = Vector2(0, 220)

	for i in range(2):
		var rect := TextureRect.new()
		rect.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		rect.size_flags_vertical   = Control.SIZE_EXPAND_FILL
		rect.stretch_mode          = TextureRect.STRETCH_KEEP_ASPECT_COVERED
		rect.expand_mode           = TextureRect.EXPAND_FIT_WIDTH_PROPORTIONAL
		if _pole_data.pics.size() > i and _pole_data.pics[i] != null:
			rect.texture = ImageTexture.create_from_image(_pole_data.pics[i])
		hbox.add_child(rect)

	var wrapper := _padded(hbox, 8, 8, 8, 8)
	field_vbox.add_child(wrapper)


# ── Build one row per field ───────────────────────────────────────────────────

func _build_field_rows() -> void:
	var confidences : Dictionary = _pole_data.confidences if _pole_data.confidences else {}

	for entry in FIELDS:
		var field_id    : String = entry[0]
		var disp_label  : String = entry[1]
		var widget_type : String = entry[2]
		var conf_key    : String = entry[3]
		var conf        : float  = float(confidences.get(conf_key, 0.0))

		_confirmed[field_id] = false

		# Outer card
		var card := PanelContainer.new()
		var card_style := StyleBoxFlat.new()
		card_style.bg_color             = Color(0.22, 0.22, 0.22, 1)
		card_style.corner_radius_top_left    = 6
		card_style.corner_radius_top_right   = 6
		card_style.corner_radius_bottom_right = 6
		card_style.corner_radius_bottom_left  = 6
		card.add_theme_stylebox_override("panel", card_style)
		card.custom_minimum_size = Vector2(0, 110)

		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 0)

		# Left confidence bar (6 px wide)
		var bar := ColorRect.new()
		bar.custom_minimum_size = Vector2(8, 0)
		bar.size_flags_vertical = Control.SIZE_EXPAND_FILL
		bar.color               = _conf_color(conf)
		_conf_bars[field_id]    = bar
		row.add_child(bar)

		# Content area
		var content := VBoxContainer.new()
		content.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		content.add_theme_constant_override("separation", 4)

		var header_row := HBoxContainer.new()
		header_row.add_theme_constant_override("separation", 8)

		var lbl := Label.new()
		lbl.text = disp_label
		lbl.add_theme_font_size_override("font_size", 30)
		lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		header_row.add_child(lbl)

		var pct_lbl := Label.new()
		pct_lbl.text = _conf_pct_text(conf)
		pct_lbl.add_theme_font_size_override("font_size", 26)
		pct_lbl.add_theme_color_override("font_color", _conf_color(conf))
		_conf_labels[field_id] = pct_lbl
		header_row.add_child(pct_lbl)

		content.add_child(header_row)

		# Input widget
		var widget : Control = _make_widget(field_id, widget_type)
		_input_nodes[field_id] = widget
		content.add_child(widget)

		var inner_pad := _padded(content, 10, 10, 10, 6)
		row.add_child(inner_pad)

		# Confirm button
		var confirm_btn := Button.new()
		confirm_btn.text                    = "✓"
		confirm_btn.custom_minimum_size     = Vector2(72, 0)
		confirm_btn.size_flags_vertical     = Control.SIZE_EXPAND_FILL
		confirm_btn.add_theme_font_size_override("font_size", 36)
		confirm_btn.pressed.connect(_on_confirm_pressed.bind(field_id))
		_confirm_btns[field_id] = confirm_btn
		row.add_child(confirm_btn)

		card.add_child(row)
		field_vbox.add_child(card)

	_total_fields = FIELDS.size()


# ── Widget factories ──────────────────────────────────────────────────────────

func _make_widget(field_id: String, widget_type: String) -> Control:
	match widget_type:
		"text":
			var te := TextEdit.new()
			te.custom_minimum_size = Vector2(0, 56)
			te.add_theme_font_size_override("font_size", 34)
			te.size_flags_horizontal = Control.SIZE_EXPAND_FILL
			match field_id:
				"pole_id":
					te.text = _pole_data.poleID
					te.placeholder_text = "Pole ID"
				"pole_type":
					te.text = _pole_data.poleType
					te.placeholder_text = "Pole Type"
			return te

		"checkbox":
			var cb := CheckBox.new()
			cb.add_theme_font_size_override("font_size", 30)
			match field_id:
				"transformer":
					cb.button_pressed = _pole_data.surroundings.has("transformer")
				"insulator":
					cb.button_pressed = _pole_data.surroundings.has("insulator")
				"streetlight":
					cb.button_pressed = _pole_data.surroundings.has("streetlight")
				"encroachment":
					cb.button_pressed = _pole_data.ai_encroachment
			return cb

		"option":
			var ob := OptionButton.new()
			ob.add_theme_font_size_override("font_size", 30)
			ob.size_flags_horizontal = Control.SIZE_EXPAND_FILL
			for opt in ["None", "Low", "Medium", "High"]:
				ob.add_item(opt)
			ob.select(_pole_data.vegetation)
			return ob

	# Fallback
	return Label.new()


# ── Helper: wrap a control in a MarginContainer with uniform padding ──────────

func _padded(node: Control, left: int, right: int, top: int, bottom: int) -> MarginContainer:
	var mc := MarginContainer.new()
	mc.add_theme_constant_override("margin_left",   left)
	mc.add_theme_constant_override("margin_right",  right)
	mc.add_theme_constant_override("margin_top",    top)
	mc.add_theme_constant_override("margin_bottom", bottom)
	mc.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	mc.add_child(node)
	return mc


# ── Confidence display helpers ────────────────────────────────────────────────

func _conf_pct_text(c: float) -> String:
	if c <= 0.0:
		return "—"
	return str(int(round(c * 100.0))) + "%"


# ── Banner ────────────────────────────────────────────────────────────────────

func _update_banner() -> void:
	var confidences : Dictionary = _pole_data.confidences if _pole_data.confidences else {}
	var has_low := false
	for entry in FIELDS:
		var conf_key : String = entry[3]
		var conf     : float  = float(confidences.get(conf_key, 0.0))
		if conf < 0.85:
			has_low = true
			break
	var banner_panel := get_node_or_null("ReviewBanner") as Panel
	if banner_panel:
		banner_panel.visible = has_low


# ── Submit label ──────────────────────────────────────────────────────────────

func _update_submit_label() -> void:
	var confirmed_count := 0
	for v in _confirmed.values():
		if v:
			confirmed_count += 1
	submit_btn.text = "Confirm All / Submit  (%d / %d)" % [confirmed_count, _total_fields]


# ── Per-field confirm button ──────────────────────────────────────────────────

func _on_confirm_pressed(field_id: String) -> void:
	_confirmed[field_id] = not _confirmed[field_id]
	var btn : Button = _confirm_btns[field_id]
	if _confirmed[field_id]:
		btn.add_theme_color_override("font_color", Color(0.2, 0.75, 0.3))
		btn.text = "✓"
	else:
		btn.remove_theme_color_override("font_color")
		btn.text = "✓"
	_update_submit_label()


# ── Collect form values back into PoleData ────────────────────────────────────

func _collect_data() -> void:
	var pd := _pole_data

	var id_te := _input_nodes.get("pole_id") as TextEdit
	if id_te:
		pd.poleID = id_te.text

	var type_te := _input_nodes.get("pole_type") as TextEdit
	if type_te:
		pd.poleType = type_te.text

	pd.surroundings = []
	for comp in ["transformer", "insulator", "streetlight"]:
		var cb := _input_nodes.get(comp) as CheckBox
		if cb and cb.button_pressed:
			pd.surroundings.append(comp)

	var veg_ob := _input_nodes.get("vegetation") as OptionButton
	if veg_ob:
		pd.vegetation = veg_ob.selected

	var enc_cb := _input_nodes.get("encroachment") as CheckBox
	if enc_cb:
		pd.ai_encroachment = enc_cb.button_pressed
	
	_pole_data = pd


# ── Submit to server ──────────────────────────────────────────────────────────

func _on_submit_pressed() -> void:
	_collect_data()
	save_data()

	var pd := _pole_data
	var payload := JSON.stringify({
		"pole_id":             pd.poleID,
		"pole_type":           pd.poleType,
		"detected_components": pd.surroundings,
		"vegetation_severity": pd.vegetation,
		"encroachment":        pd.ai_encroachment,
	})

	submit_btn.disabled = true
	submit_btn.text     = "Submitting…"

	var headers : PackedStringArray = ["Content-Type: application/json"]
	var err : int = _http.request(SUBMIT_URL, headers, HTTPClient.METHOD_POST, payload)
	if err != OK:
		_show_submit_error("Could not reach server (error %d)." % err)


func _on_submit_completed(
	result        : int,
	response_code : int,
	_headers      : PackedStringArray,
	_body         : PackedByteArray,
) -> void:
	if result != HTTPRequest.RESULT_SUCCESS or response_code != 200:
		_show_submit_error("Server returned HTTP %d." % response_code)
		return

	# Navigate back to the main menu after a successful submission
	var main_scene = load("res://screens/main.tscn")
	if main_scene:
		get_tree().root.add_child(main_scene.instantiate())
	queue_free()


func _show_submit_error(msg: String) -> void:
	submit_btn.disabled = false
	submit_btn.text     = "Confirm All / Submit  (retry)"
	# Surface the error in the banner label so it's visible
	if banner_label:
		banner_label.text = "Submit failed: " + msg
	var banner_panel := get_node_or_null("ReviewBanner") as Panel
	if banner_panel:
		banner_panel.visible = true
