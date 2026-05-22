from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator, QIntValidator, QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QGroupBox,
    QScrollArea,
    QFrame,
)

class MainContentWidget(QWidget):
    batch_requested = Signal()
    predict_requested = Signal(dict)
    search_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.inputs = {}
        self.init_ui()

    def add_input_field(self, layout, label_text, key, field_type="text", items=None, is_numeric=True):
        # Use a bold label for the form field
        label = QLabel(f"<b>{label_text}:</b>")
        
        if field_type == "combo":
            widget = QComboBox()
            widget.addItems(items or [])
        else:
            widget = QLineEdit()
            if is_numeric:
                # Restricts input to standard notation
                validator = QDoubleValidator(0.0, 999999.99, 2)
                validator.setNotation(QDoubleValidator.StandardNotation)
                widget.setValidator(validator)

        self.inputs[key] = widget
        layout.addWidget(label)
        layout.addWidget(widget)

    def init_ui(self):
        self.setWindowTitle("Credit Customer Churn Prediction")
        self.resize(600, 800)
        
        main_layout = QVBoxLayout(self)

        # HEADER
        header = QLabel("Customer Churn Analysis & Retention")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        # SEARCH SECTION
        search_group = QGroupBox("Search for Customer")
        search_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter Client ID")
        self.input_field.setValidator(QIntValidator())
        
        srch_btn = QPushButton("Search")
        srch_btn.setStyleSheet("background-color: #2980B9; color: white; padding: 5px;")
        self.input_field.returnPressed.connect(self.emit_search)
        srch_btn.clicked.connect(self.emit_search)

        search_layout.addWidget(self.input_field)
        search_layout.addWidget(srch_btn)
        search_group.setLayout(search_layout)
        main_layout.addWidget(search_group)

        # SCROLL AREA
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        scroll_layout = QVBoxLayout(content_widget)

        # Account Section
        acc_group = QGroupBox("Account & Demographics")
        acc_lay = QVBoxLayout(acc_group)
        account_fields = [
            ("Dependent Count", "dependent_count", "text", None),
            ("Education Level", "education_level", "combo", ["Uneducated", "High School", "College", "Graduate", "Doctorate", "Unknown"]),
            ("Marital Status", "marital_status", "combo", ["Single", "Married", "Divorced", "Unknown"]),
            ("Income Category", "income_category", "combo", ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"]),
            ("Card Category", "card_category", "combo", ["Blue", "Silver", "Gold", "Platinum"])
        ]
        for label, key, f_type, items in account_fields:
            self.add_input_field(acc_lay, label, key, f_type, items)
        scroll_layout.addWidget(acc_group)

        # Financial Section
        fin_group = QGroupBox("Financial Activity")
        fin_lay = QVBoxLayout(fin_group)
        financial_fields = [
            ("Months on Book", "months_on_book"),
            ("Total Relationship Count", "total_relationship_count"),
            ("Months Inactive 12 Mon", "months_inactive_12_mon"),
            ("Contacts Count 12 Mon", "contacts_count_12_mon"),
            ("Credit Limit ($)", "credit_limit"),
            ("Total Revolving Bal ($)", "total_revolving_bal"),
            ("Total Amt Chng Q4-Q1", "total_amt_chng_q4_q1"),
            ("Total Trans Amt ($)", "total_trans_amt"),
            ("Total Trans Ct", "total_trans_ct"),
            ("Total Ct Chng Q4-Q1", "total_ct_chng_q4_q1"),
            ("Avg Utilization Ratio", "avg_utilization_ratio")
        ]
        for label, key in financial_fields:
            self.add_input_field(fin_lay, label, key)
        
        scroll_layout.addWidget(fin_group)
        scroll_layout.addStretch()
        content_widget.setLayout(scroll_layout)
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

        # ACTION BUTTONS
        btn_layout = QHBoxLayout()
        
        self.predict_btn = QPushButton("Calculate Churn Risk")
        self.predict_btn.setStyleSheet("background-color: #2E86C1; color: white; padding: 10px; font-weight: bold;")
        self.predict_btn.clicked.connect(self.emit_predict)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet("background-color: #E74C3C; color: white; padding: 10px; font-weight: bold;")
        self.clear_btn.clicked.connect(self.clear_inputs)

        self.batch_btn = QPushButton("Run Batch Analysis")
        self.batch_btn.setStyleSheet("background-color: #27AE60; color: white; padding: 10px; font-weight: bold;")
        self.batch_btn.clicked.connect(self.batch_requested.emit)

        btn_layout.addWidget(self.predict_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.batch_btn)
        main_layout.addLayout(btn_layout)

        # RESULTS SECTION
        self.result_label = QLabel("Churn Risk: -- | Stay Probability: --")
        self.result_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)

        analysis_lay = QHBoxLayout()
        self.factors_box = QLabel("<b>Key Drivers & Strengths:</b><br><i>Pending...</i>")
        self.incentives_box = QLabel("<b>Actionable Incentives:</b><br><i>Pending...</i>")
        for box in [self.factors_box, self.incentives_box]:
            box.setFrameStyle(QFrame.Panel | QFrame.Sunken)
            box.setWordWrap(True)
            box.setMinimumHeight(150)
            analysis_lay.addWidget(box)
        main_layout.addLayout(analysis_lay)

    def emit_search(self):
        client_id = self.input_field.text().strip()
        if client_id:
            self.search_requested.emit(client_id)

    def emit_predict(self):
        self.predict_requested.emit(self.get_form_data())

    def populate_fields(self, data):
        clean_data = {k.lower(): v for k, v in data.items()}
        for key, widget in self.inputs.items():
            value = clean_data.get(key.lower())
            if value is not None:
                if isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                elif isinstance(widget, QComboBox):
                    index = widget.findText(str(value), Qt.MatchExactly)
                    if index >= 0:
                        widget.setCurrentIndex(index)

    def get_form_data(self):
        return {k: (w.currentText() if isinstance(w, QComboBox) else w.text()) 
                for k, w in self.inputs.items()}

    def clear_inputs(self):
        self.input_field.clear()
        for w in self.inputs.values():
            w.setCurrentIndex(0) if isinstance(w, QComboBox) else w.clear()
        self.result_label.setText("Churn Risk: -- | Stay Probability: --")
        self.result_label.setStyleSheet("")
        self.factors_box.setText("<b>Key Drivers & Strengths:</b><br><i>Pending...</i>")
        self.incentives_box.setText("<b>Actionable Incentives:</b><br><i>Pending...</i>")

    def update_ui_with_results(self, res):
        churn_risk = res.get('probability', 0) * 100
        stay_prob = max(0.0, 100 - churn_risk)
        status_text = res.get('status', 'Unknown')
        
        color = '#C0392B' if churn_risk > 50 else '#27AE60'
        self.result_label.setText(f"<b>{status_text}</b> | Risk: {churn_risk:.1f}% | Stay: {stay_prob:.1f}%")
        self.result_label.setStyleSheet(f"color: {color};")

        factors = res.get("top_factors", [])
        drivers = [f"<span style='color: #C0392B;'>• {f['feature']} (+{f['impact']:.1f}%)</span>" 
                   for f in factors if f['impact'] > 0]
        strengths = [f"<span style='color: #27AE60;'>• {f['feature']} ({f['impact']:.1f}%)</span>" 
                     for f in factors if f['impact'] <= 0]

        driver_html = "<b>Risk Drivers:</b><br>" + ("<br>".join(drivers) if drivers else "<i>None</i>")
        strength_html = "<b>Retention Strengths:</b><br>" + ("<br>".join(strengths) if strengths else "<i>None</i>")
        self.factors_box.setText(strength_html + "<br><br>" + driver_html)

        # Simplified logic for incentives based on top feature
        strategy = "• Schedule Account Review"
        if factors:
            top_f = factors[0]['feature'].lower()
            if "limit" in top_f: strategy = "• Credit Line Evaluation"
            elif "trans" in top_f: strategy = "• Loyalty Reward Program"
            elif "inactive" in top_f: strategy = "• Re-engagement Campaign"

        self.incentives_box.setText(f"<b>Retention Strategy:</b><br>{strategy}")