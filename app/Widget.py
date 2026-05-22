import sys
import requests

from PySide6.QtWidgets import QApplication, QMessageBox, QWidget, QTabWidget, QVBoxLayout
from PySide6.QtCore import Qt
from app.background.batch import BatchResultsTable
from app.background.ui_content import MainContentWidget

class ChurnPredictionWidget(QWidget):
    def __init__(self):
        super().__init__()
        # Keep one backend URL for all requests.
        self.api_base_url = "http://localhost:8000"
        self.setWindowTitle("Churn Analysis Dashboard")
        self.resize(1000, 800)

        # Setup Tabs
        self.tabs = QTabWidget()
        self.prediction_tab = MainContentWidget()
        self.batch_container = QWidget()
        self.batch_layout = QVBoxLayout(self.batch_container)
        
        self.tabs.addTab(self.prediction_tab, "Single Analysis")
        self.tabs.addTab(self.batch_container, "Batch Insights")

        # Main Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

        self.prediction_tab.search_requested.connect(self.execute_search)
        self.prediction_tab.predict_requested.connect(self.execute_prediction)
        self.prediction_tab.batch_requested.connect(self.run_batch_process)
        self.prediction_tab.clear_btn.clicked.connect(self.prediction_tab.clear_inputs)

    def execute_search(self, client_id):
        if not client_id: return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            response = requests.get(f"{self.api_base_url}/customer/{client_id}", timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                self.prediction_tab.populate_fields(data.get("data"))
            else:
                QMessageBox.warning(self, "Search", "Customer not found.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Search failed: {str(e)}")
        
        finally:
            QApplication.restoreOverrideCursor()

    def execute_prediction(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        input_data = self.prediction_tab.get_form_data()

        self.prediction_tab.predict_btn.setEnabled(False)
        self.prediction_tab.predict_btn.setText("Processing...")
        QApplication.processEvents()
        try:
            response = requests.post(f"{self.api_base_url}/predict", json=input_data, timeout=10)
            response.raise_for_status()
            
            res = response.json().get("prediction", {})
            self.prediction_tab.update_ui_with_results(res)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")
        
        finally:
            QApplication.restoreOverrideCursor()
            self.prediction_tab.predict_btn.setEnabled(True)
            self.prediction_tab.predict_btn.setText("Calculate Churn Risk")

    def run_batch_process(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.prediction_tab.batch_btn.setEnabled(False)
        self.prediction_tab.batch_btn.setText("Processing...")
        QApplication.processEvents()
        try:
            response = requests.get(f"{self.api_base_url}/batch_predict", timeout=30)
            response.raise_for_status()

            self.handle_batch_finished(response.json())

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Batch failed: {str(e)}")
        
        finally:
            self.prediction_tab.batch_btn.setText("Run Batch Analysis")
            QApplication.restoreOverrideCursor()

    def handle_batch_finished(self, data): 
        QApplication.restoreOverrideCursor()
        self.prediction_tab.batch_btn.setEnabled(True)
        self.prediction_tab.batch_btn.setText("Run Batch Analysis")

        for i in reversed(range(self.batch_layout.count())): 
            widget = self.batch_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        results = data.get('results', [])
        table = BatchResultsTable(results) 
        table.customer_selected.connect(self.go_to_customer)
        self.batch_layout.addWidget(table)
        self.tabs.setCurrentIndex(1)

    def go_to_customer(self, client_id):
        self.prediction_tab.input_field.setText(client_id)
        self.tabs.setCurrentIndex(0)
        self.execute_search(client_id)




if __name__ == "__main__":
     app = QApplication(sys.argv)
     window = ChurnPredictionWidget()
     window.show()
     sys.exit(app.exec())

# Usage:
# python -m app.Widget