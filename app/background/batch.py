
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QCursor, QFont
from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem

class BatchResultsTable(QTableWidget):
    customer_selected = Signal(str)

    def __init__(self, results):
        super().__init__(len(results), 2)
        self.setHorizontalHeaderLabels(["Client Number", "Churn Probability"])
        
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSelectionMode(QTableWidget.SingleSelection)

        # Signals
        self.itemClicked.connect(self.on_item_clicked)

        if results:
            self.populate_data(results)

    def populate_data(self, results):
        # Sort results by probability descending
        sorted_results = sorted(results, key=lambda x: x.get('probability', 0), reverse=True)

        for i, res in enumerate(sorted_results):
            # Client Number
            client_id = str(res.get('clientnum', 'Unknown'))
            client_item = QTableWidgetItem(client_id)
            
            # Professional link-style formatting
            font = QFont()
            font.setUnderline(True)
            client_item.setFont(font)
            client_item.setForeground(QColor("#2980B9")) 
            client_item.setTextAlignment(Qt.AlignCenter)
            self.setItem(i, 0, client_item)
        
            # Churn Percentage
            prob = res.get('probability', 0) * 100
            prob_item = QTableWidgetItem(f"{prob:.1f}%")
            prob_item.setTextAlignment(Qt.AlignCenter) 
            
            # Conditional text color based on risk level
            if prob > 50:
                prob_item.setForeground(QColor("#C0392B")) # Red for high risk
                font = QFont()
                font.setBold(True)
                prob_item.setFont(font)
            self.setItem(i, 1, prob_item)

        self.viewport().setCursor(QCursor(Qt.PointingHandCursor))
        
    def on_item_clicked(self, item):
        # Only navigate if the user clicks the Client ID column
        if item.column() == 0:
            self.customer_selected.emit(item.text())