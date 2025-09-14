// BCC Bank Personalization System
class PersonalizationApp {
    constructor() {
        this.clients = [];
        this.recommendations = [];
        this.currentSection = 'dashboard';
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadClients();
        this.setupNavigation();
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = e.target.getAttribute('href').substring(1);
                this.showSection(section);
            });
        });

        // Generate all recommendations
        document.getElementById('generate-all-btn').addEventListener('click', () => {
            this.generateAllRecommendations();
        });

        // Export CSV
        document.getElementById('export-csv-btn').addEventListener('click', () => {
            this.exportCSV();
        });

        // Client search
        document.getElementById('client-search').addEventListener('input', (e) => {
            this.filterClients(e.target.value);
        });

        // Status filter
        document.getElementById('status-filter').addEventListener('change', (e) => {
            this.filterClientsByStatus(e.target.value);
        });

        // Modal close
        document.querySelector('.close').addEventListener('click', () => {
            this.closeModal();
        });

        // ML Controls
        document.getElementById('init-ml-btn').addEventListener('click', () => {
            this.initML();
        });

        document.getElementById('ml-recommendations-btn').addEventListener('click', () => {
            this.loadMLRecommendations();
        });

        document.getElementById('ml-export-btn').addEventListener('click', () => {
            this.exportMLCSV();
        });

        // Close modal on outside click
        document.getElementById('client-modal').addEventListener('click', (e) => {
            if (e.target.id === 'client-modal') {
                this.closeModal();
            }
        });
    }

    setupNavigation() {
        // Set active nav link based on current section
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${this.currentSection}`) {
                link.classList.add('active');
            }
        });
    }

    showSection(sectionName) {
        // Hide all sections
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });

        // Show selected section
        document.getElementById(sectionName).classList.add('active');
        this.currentSection = sectionName;

        // Update navigation
        this.setupNavigation();

        // Load section-specific data
        if (sectionName === 'analytics') {
            this.loadAnalytics();
        }
    }

    async loadClients() {
        try {
            this.showLoading();
            const response = await fetch('/api/clients');
            this.clients = await response.json();
            this.renderClientsTable();
            this.updateStats();
        } catch (error) {
            console.error('Ошибка загрузки клиентов:', error);
            this.showError('Ошибка загрузки данных клиентов');
        } finally {
            this.hideLoading();
        }
    }

    renderClientsTable() {
        const tbody = document.getElementById('clients-table-body');
        tbody.innerHTML = '';

        this.clients.forEach(client => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${client.client_code}</td>
                <td>${client.name}</td>
                <td><span class="status-badge status-${this.getStatusClass(client.status)}">${client.status}</span></td>
                <td>${client.age}</td>
                <td>${client.city}</td>
                <td>${this.formatCurrency(client.avg_monthly_balance_KZT)}</td>
                <td><button class="analyze-btn" onclick="app.analyzeClient(${client.client_code})">Анализ</button></td>
            `;
            tbody.appendChild(row);
        });
    }

    getStatusClass(status) {
        const statusMap = {
            'Студент': 'student',
            'Зарплатный клиент': 'salary',
            'Премиальный клиент': 'premium',
            'Стандартный клиент': 'standard'
        };
        return statusMap[status] || 'standard';
    }

    filterClients(searchTerm) {
        const rows = document.querySelectorAll('#clients-table-body tr');
        rows.forEach(row => {
            const text = row.textContent.toLowerCase();
            const matches = text.includes(searchTerm.toLowerCase());
            row.style.display = matches ? '' : 'none';
        });
    }

    filterClientsByStatus(status) {
        const rows = document.querySelectorAll('#clients-table-body tr');
        rows.forEach(row => {
            if (!status) {
                row.style.display = '';
                return;
            }
            const statusCell = row.querySelector('.status-badge');
            const matches = statusCell && statusCell.textContent === status;
            row.style.display = matches ? '' : 'none';
        });
    }

    async analyzeClient(clientCode) {
        try {
            this.showLoading();
            const response = await fetch(`/api/analyze/${clientCode}`);
            const analysis = await response.json();
            this.showClientModal(analysis);
        } catch (error) {
            console.error('Ошибка анализа клиента:', error);
            this.showError('Ошибка анализа клиента');
        } finally {
            this.hideLoading();
        }
    }

    showClientModal(analysis) {
        const modal = document.getElementById('client-modal');
        const title = document.getElementById('modal-title');
        const body = document.getElementById('modal-body');

        title.textContent = `Анализ клиента ${analysis.client_code}`;
        
        body.innerHTML = `
            <div class="client-analysis">
                <div class="analysis-header">
                    <h3>Рекомендация</h3>
                    <div class="product-badge">${analysis.product}</div>
                </div>
                
                <div class="benefit-info">
                    <h4>Ожидаемая выгода: <span class="benefit-amount">${this.formatCurrency(analysis.benefit)}</span></h4>
                    <p>${analysis.reason}</p>
                </div>
                
                <div class="push-message">
                    <h4>Пуш-уведомление:</h4>
                    <p>"${analysis.push_notification}"</p>
                </div>
            </div>
        `;

        modal.style.display = 'block';
    }

    closeModal() {
        document.getElementById('client-modal').style.display = 'none';
    }

    async generateAllRecommendations() {
        try {
            this.showLoading();
            const response = await fetch('/api/generate_all');
            this.recommendations = await response.json();
            this.renderRecommendations();
            this.updateStats();
        } catch (error) {
            console.error('Ошибка генерации рекомендаций:', error);
            this.showError('Ошибка генерации рекомендаций');
        } finally {
            this.hideLoading();
        }
    }

    renderRecommendations() {
        const container = document.getElementById('recommendations-list');
        container.innerHTML = '';

        if (this.recommendations.length === 0) {
            container.innerHTML = '<p class="text-center">Рекомендации не сгенерированы</p>';
            return;
        }

        this.recommendations.forEach(rec => {
            const card = document.createElement('div');
            card.className = 'recommendation-card';
            card.innerHTML = `
                <div class="recommendation-header">
                    <div class="client-info">
                        <div class="client-avatar">${rec.client_code}</div>
                        <div class="client-details">
                            <h4>Клиент #${rec.client_code}</h4>
                            <p>Рекомендуемый продукт</p>
                        </div>
                    </div>
                    <div class="product-badge">${rec.product}</div>
                </div>
                
                <div class="benefit-amount">${this.formatCurrency(rec.benefit)}</div>
                <p class="mb-2">${rec.reason}</p>
                
                <div class="push-message">
                    <strong>Пуш-уведомление:</strong><br>
                    "${rec.push_notification}"
                </div>
            `;
            container.appendChild(card);
        });
    }

    async exportCSV() {
        try {
            this.showLoading();
            const response = await fetch('/api/export_csv');
            const blob = await response.blob();
            
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'recommendations.csv';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (error) {
            console.error('Ошибка экспорта:', error);
            this.showError('Ошибка экспорта CSV');
        } finally {
            this.hideLoading();
        }
    }

    loadAnalytics() {
        this.renderStatusChart();
        this.renderCategoriesChart();
    }

    renderStatusChart() {
        const ctx = document.getElementById('status-chart').getContext('2d');
        
        // Count clients by status
        const statusCounts = {};
        this.clients.forEach(client => {
            statusCounts[client.status] = (statusCounts[client.status] || 0) + 1;
        });

        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(statusCounts),
                datasets: [{
                    data: Object.values(statusCounts),
                    backgroundColor: [
                        '#3b82f6',
                        '#10b981',
                        '#f59e0b',
                        '#ef4444'
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    renderCategoriesChart() {
        const ctx = document.getElementById('categories-chart').getContext('2d');
        
        // Mock data for categories (in real app, this would come from API)
        const categoryData = {
            labels: ['Продукты питания', 'Кафе и рестораны', 'Такси', 'Онлайн-сервисы', 'Путешествия'],
            datasets: [{
                label: 'Сумма трат (₸)',
                data: [1200000, 800000, 400000, 300000, 200000],
                backgroundColor: [
                    '#3b82f6',
                    '#10b981',
                    '#f59e0b',
                    '#ef4444',
                    '#8b5cf6'
                ]
            }]
        };

        new Chart(ctx, {
            type: 'bar',
            data: categoryData,
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return value.toLocaleString() + ' ₸';
                            }
                        }
                    }
                }
            }
        });
    }

    updateStats() {
        document.getElementById('total-clients').textContent = this.clients.length;
        document.getElementById('total-recommendations').textContent = this.recommendations.length;
        
        if (this.recommendations.length > 0) {
            const avgBenefit = this.recommendations.reduce((sum, rec) => sum + rec.benefit, 0) / this.recommendations.length;
            document.getElementById('avg-benefit').textContent = this.formatCurrency(avgBenefit);
            document.getElementById('push-ready').textContent = this.recommendations.length;
        }
    }

    formatCurrency(amount) {
        return new Intl.NumberFormat('ru-RU', {
            style: 'currency',
            currency: 'KZT',
            minimumFractionDigits: 0
        }).format(amount);
    }

    showLoading() {
        document.getElementById('loading-overlay').style.display = 'block';
    }

    hideLoading() {
        document.getElementById('loading-overlay').style.display = 'none';
    }

    showError(message) {
        alert(message);
    }

    // ML Methods
    async initML() {
        this.showLoading();
        
        try {
            const response = await fetch('/api/ml/init');
            const data = await response.json();
            
            if (data.success) {
                this.showMLStatus(data);
                this.enableMLButtons();
                this.loadMLInsights();
                this.showNotification('ML анализатор успешно инициализирован!', 'success');
            } else {
                this.showNotification('Ошибка инициализации ML: ' + data.error, 'error');
            }
        } catch (error) {
            this.showNotification('Ошибка подключения к серверу', 'error');
        } finally {
            this.hideLoading();
        }
    }

    showMLStatus(data) {
        const statusDiv = document.getElementById('ml-status');
        const contentDiv = document.getElementById('ml-status-content');
        
        contentDiv.innerHTML = `
            <div class="status-item">
                <span class="status-label">Точность модели:</span>
                <span class="status-value">${(data.accuracy * 100).toFixed(1)}%</span>
            </div>
            <div class="status-item">
                <span class="status-label">Количество признаков:</span>
                <span class="status-value">${data.features_count}</span>
            </div>
            <div class="status-item">
                <span class="status-label">Кластеров обнаружено:</span>
                <span class="status-value">${data.clusters_count}</span>
            </div>
        `;
        
        statusDiv.style.display = 'block';
    }

    enableMLButtons() {
        document.getElementById('ml-recommendations-btn').disabled = false;
        document.getElementById('ml-export-btn').disabled = false;
    }

    async loadMLInsights() {
        try {
            const response = await fetch('/api/ml/insights');
            const insights = await response.json();
            
            this.displayClustersInfo(insights);
        } catch (error) {
            console.error('Ошибка загрузки инсайтов:', error);
        }
    }

    displayClustersInfo(insights) {
        const clustersDiv = document.getElementById('clusters-info');
        
        let html = '<div class="clusters-grid">';
        insights.forEach(cluster => {
            html += `
                <div class="cluster-card">
                    <h4>${cluster.cluster_type}</h4>
                    <div class="cluster-stats">
                        <div class="stat">
                            <span class="label">Размер:</span>
                            <span class="value">${cluster.size} клиентов</span>
                        </div>
                        <div class="stat">
                            <span class="label">Возраст:</span>
                            <span class="value">${cluster.avg_age.toFixed(1)} лет</span>
                        </div>
                        <div class="stat">
                            <span class="label">Баланс:</span>
                            <span class="value">${cluster.avg_balance.toLocaleString()} ₸</span>
                        </div>
                        <div class="stat">
                            <span class="label">Путешествия:</span>
                            <span class="value">${(cluster.characteristics.travel_ratio * 100).toFixed(1)}%</span>
                        </div>
                        <div class="stat">
                            <span class="label">Онлайн:</span>
                            <span class="value">${(cluster.characteristics.online_ratio * 100).toFixed(1)}%</span>
                        </div>
                        <div class="stat">
                            <span class="label">Люкс:</span>
                            <span class="value">${(cluster.characteristics.luxury_ratio * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                    <div class="recommended-products">
                        <strong>Рекомендуемые продукты:</strong>
                        <div class="product-tags">
                            ${cluster.recommended_products.map(product => `<span class="product-tag">${product}</span>`).join('')}
                        </div>
                    </div>
                </div>
            `;
        });
        html += '</div>';
        
        clustersDiv.innerHTML = html;
    }

    async loadMLRecommendations() {
        this.showLoading();
        
        try {
            const response = await fetch('/api/ml/recommendations');
            const recommendations = await response.json();
            
            this.displayMLRecommendations(recommendations);
            this.showNotification(`Сгенерировано ${recommendations.length} ML рекомендаций`, 'success');
        } catch (error) {
            this.showNotification('Ошибка загрузки ML рекомендаций', 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayMLRecommendations(recommendations) {
        const container = document.getElementById('ml-recommendations-list');
        
        if (recommendations.length === 0) {
            container.innerHTML = '<p class="no-data">Нет рекомендаций</p>';
            return;
        }
        
        let html = '';
        recommendations.forEach(rec => {
            const confidenceClass = rec.confidence > 0.8 ? 'high' : rec.confidence > 0.6 ? 'medium' : 'low';
            
            html += `
                <div class="recommendation-card ml-card">
                    <div class="rec-header">
                        <div class="client-info">
                            <h4>${rec.name}</h4>
                            <span class="client-id">ID: ${rec.client_code}</span>
                        </div>
                        <div class="rec-meta">
                            <span class="confidence confidence-${confidenceClass}">
                                ${(rec.confidence * 100).toFixed(1)}% уверенность
                            </span>
                            <span class="cluster-badge">Кластер ${rec.cluster}</span>
                        </div>
                    </div>
                    <div class="rec-content">
                        <div class="product-info">
                            <h5>Рекомендуемый продукт: ${rec.product}</h5>
                            <p class="benefit">Ожидаемая выгода: ${rec.benefit.toLocaleString()} ₸</p>
                        </div>
                        <div class="push-message">
                            <strong>Пуш-уведомление:</strong>
                            <p>${rec.push_notification}</p>
                        </div>
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
    }

    async exportMLCSV() {
        try {
            const response = await fetch('/api/ml/export');
            const blob = await response.blob();
            
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'ml_recommendations.csv';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            this.showNotification('ML рекомендации экспортированы в CSV', 'success');
        } catch (error) {
            this.showNotification('Ошибка экспорта ML данных', 'error');
        }
    }

    showNotification(message, type = 'info') {
        // Создаем уведомление
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Добавляем стили
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 5px;
            color: white;
            font-weight: bold;
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;
        
        if (type === 'success') {
            notification.style.backgroundColor = '#10b981';
        } else if (type === 'error') {
            notification.style.backgroundColor = '#ef4444';
        } else {
            notification.style.backgroundColor = '#3b82f6';
        }
        
        document.body.appendChild(notification);
        
        // Удаляем через 3 секунды
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new PersonalizationApp();
});
