import dearpygui.dearpygui as dpg

import subprocess

import os
script_dir = os.path.dirname( os.path.abspath(__file__) )

models = ["Modèle rule-based naïf", "Modèle rule-based amélioré", "Modèle rule-based de INCLURE", "Modèle statistique avec CRF", "Modèle de vote"]
selected = None
sub_selection = False
sub_selected = []
sub_weights = []
tol = None
selected_id = None

def select_model(button, app_data, cbbox) :
	global selected, selected_id, sub_selection
	tmp = models.index(dpg.get_value(cbbox))

	if sub_selection is not None:
		if sub_selection:
			ext = "_sub"
		else :
			ext = ""
		dpg.delete_item("opts" + ext)

	selected_id = tmp
	add_options()

def add_to_list(item, app_data, user_data) :
	val = dpg.get_item_label(item)
	if val in user_data :
		user_data.remove(val)
	else:
		user_data.append(val)


def validate_model(button, app_data, user_data) :
	global selected, selected_id, sub_selected, sub_selection, sub_weights, tol
	match selected_id :
		case 0 :
			tmp = "NaiveRBModel()"
		case 1 :
			tmp = "AdvRBModel(proc=" + str(user_data["proc"]) + ")"
		case 2 :
			tmp = "InclureModel(proc=" + str(user_data["proc"]) + ")"
		case 3 :
			tmp = "CRFModel()"
		case 4 :
			if len(sub_selected) > 0 and tol is not None: 
				mods = "["
				for item in sub_selected :
					mods += item + ","
				mods = mods[:-1] + "]"
				tmp = "SuperAnnModel(models=" + mods + ", weights =" + str(sub_weights) + ", tol =" + str(tol) + ")"
			else : 
				tmp = None
				pass
		case _ :
			pass
	if sub_selection :
		sub_selected.append(tmp)
		add_sub_weights()
	else :
		if tmp is not None :
			selected = tmp
			explore()

def explore():
	global selected
	if dpg.does_alias_exist("opts") :
		dpg.delete_item("opts")
	with dpg.window(label="Traitement") :
		dpg.add_text("model selected : " + selected)
		with open(os.path.dirname(os.path.abspath(__file__)) + "/mem.txt", "a") as f :
			f.write(selected + '\n')
		subprocess.run(["python3", script_dir+"/collect_oscar.py", selected])

def add_tol_callback(input, app_data, user_data) :
	global tol
	tol = dpg.get_value(input)
		

def add_sub_weights_callback(button, app_data, user_data) :
	global sub_weights, models, sub_selection, selected_id
	sub_weights.append(dpg.get_value(user_data))
	dpg.delete_item("popup")
	dpg.delete_item("opts_sub")
	dpg.delete_item("cb_sub")
	dpg.delete_item("sel_but_sub")
	dpg.delete_item("opts_child")
	
	dpg.add_text(models[selected_id] + " x" + str(sub_weights[-1]) , parent="opts", before="val_but")

	selected_id = 4
	sub_selection = False


def add_sub_weights() :
	with dpg.child_window( parent="opts_sub", tag="popup" ) :
		dpg.add_input_float(label="Entrer un poids : ", tag="input_weight")
		dpg.add_button(label="Valider", callback=add_sub_weights_callback, user_data="input_weight")


def add_sub_model(button, app_data, user_data) :
	global sub_selection
	if not sub_selection :
		sub_selection = True
		with dpg.child_window(parent="opts", tag="opts_child") :
			model_listing(with_vote=False)

def add_options() :
	global selected, selected_id, sub_selection, sub_selected, sub_weights

	if not sub_selection:
		sub_selected = []
		sub_weights = []

	tag ="opts" + ( "_sub" if sub_selection else "" )


	if selected_id is not None :
		dpg.add_window(label="Options", tag=tag, width=500, height=300, pos=(100,100 * (2 if sub_selection else 1) ) )
		data = {}
		match selected_id :
			case 0:
				pass
			case 1:
				data["proc"] = []
				with dpg.group(horizontal=True, parent=tag) :
					dpg.add_checkbox(label="fle", callback=add_to_list, user_data=data["proc"], parent=tag)
					dpg.add_checkbox(label="fem", callback=add_to_list, user_data=data["proc"], parent=tag)
					dpg.add_checkbox(label="coo", callback=add_to_list, user_data=data["proc"], parent=tag)
					dpg.add_checkbox(label="epi", callback=add_to_list, user_data=data["proc"], parent=tag)
					dpg.add_checkbox(label="neu", callback=add_to_list, user_data=data["proc"], parent=tag)
			case 2:
				data["proc"] = []
				with dpg.group(horizontal=True, parent=tag) :
					dpg.add_checkbox(label="fle", callback=add_to_list, user_data=data["proc"], parent=tag)
					dpg.add_checkbox(label="coo", callback=add_to_list, user_data=data["proc"], parent=tag)
			case 3:
				pass
			case 4:
				dpg.add_button(label="Nouveau sous modèle", callback=add_sub_model, parent=tag)
				dpg.add_input_float(label="Entrer une tolérance : ", tag="input", callback=add_tol_callback, on_enter=True, parent=tag)
			case _ :
				dpg.add_text("Erreur", parent=tag)

		dpg.add_button(label="Valider", callback=validate_model, user_data=data, parent=tag, tag="val_but" + ( "_sub" if sub_selection else "" ) ) 
		

def model_listing(with_vote=True) :
	if with_vote :
		cbbox = dpg.add_combo(items=models, tag="cb")
		dpg.add_button(label="Selectionner", callback=select_model, user_data=cbbox, tag="sel_but")
	else :
		cbbox_sub = dpg.add_combo(items=models[:-1], parent="opts_child", tag="cb_sub")
		dpg.add_button(label="Selectionner", callback=select_model, user_data=cbbox_sub, tag="sel_but_sub", parent="opts_child")

def launch_previous_selec(button, app_data, user_data) :
	global selected
	selected = dpg.get_value(user_data)
	dpg.delete_item("prev_sel")
	explore()

def choose_previous_selec() :
	dpg.add_window(label="Ancienne(s) selection(s)", tag="prev_sel")
	with  open(os.path.dirname(os.path.abspath(__file__)) + "/mem.txt", 'r') as f :
		dpg.add_combo(items=f.read().split('\n'), parent="prev_sel", tag="prev_sel_combo")
	dpg.add_button(label="Selectionner", parent="prev_sel", callback=launch_previous_selec, user_data="prev_sel_combo")



dpg.create_context()
dpg.create_viewport(title='Mon application', width=1020, height=800)

with dpg.window(label="Choose a model", tag="Primary_Window") as window:
	if os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/mem.txt") :
		dpg.add_button(label="Ancienne(s) selection(s)", callback=choose_previous_selec)

	model_listing()



dpg.setup_dearpygui()
dpg.set_primary_window("Primary_Window", True)
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
