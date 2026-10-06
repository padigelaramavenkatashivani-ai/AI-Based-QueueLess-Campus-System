from datetime import datetime, timezone

import numpy as np

from flask import (
    Flask,
    request,
    jsonify,
    render_template
)

from flask_sqlalchemy import SQLAlchemy

from sklearn.ensemble import RandomForestRegressor


# =========================================================
# Flask Configuration
# =========================================================

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///queueless.db"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# =========================================================
# Database Models
# =========================================================

class Service(db.Model):

    __tablename__ = "services"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(100),
        nullable=False,
        unique=True
    )

    avg_service_time = db.Column(
        db.Float,
        nullable=False,
        default=5.0
    )

    active = db.Column(
        db.Boolean,
        default=True,
        nullable=False
    )


class QueueToken(db.Model):

    __tablename__ = "queue_tokens"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    token_number = db.Column(
        db.Integer,
        nullable=False
    )

    student_name = db.Column(
        db.String(100),
        nullable=False
    )

    student_id = db.Column(
        db.String(100),
        nullable=False
    )

    service_id = db.Column(
        db.Integer,
        db.ForeignKey("services.id"),
        nullable=False
    )

    status = db.Column(
        db.String(30),
        nullable=False,
        default="waiting"
    )

    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )

    called_at = db.Column(
        db.DateTime(timezone=True),
        nullable=True
    )

    completed_at = db.Column(
        db.DateTime(timezone=True),
        nullable=True
    )

    service = db.relationship(
        "Service",
        backref="tokens"
    )


# =========================================================
# AI Waiting Time Predictor
# =========================================================

class WaitingTimePredictor:

    def __init__(self):

        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        self.trained = False

    def train(self):

        """
        Demo training dataset.

        Features:
        1. People waiting
        2. Average service time
        3. Active counters

        Target:
        Waiting time in minutes
        """

        X = np.array([

            [1, 3, 1],
            [2, 4, 1],
            [3, 5, 1],
            [5, 5, 1],
            [8, 5, 1],

            [2, 3, 2],
            [5, 4, 2],
            [8, 4, 2],

            [10, 5, 2],
            [15, 5, 3],

            [20, 5, 3],
            [25, 6, 3]

        ])

        y = np.array([

            3,
            8,
            15,
            25,
            40,

            4,
            10,
            16,

            25,
            30,

            35,
            45

        ])

        self.model.fit(
            X,
            y
        )

        self.trained = True

    def predict(
        self,
        people_waiting,
        avg_service_time,
        active_counters=1
    ):

        if not self.trained:

            self.train()

        features = np.array([[

            people_waiting,
            avg_service_time,
            active_counters

        ]])

        prediction = self.model.predict(
            features
        )[0]

        return max(
            1,
            round(float(prediction), 1)
        )


predictor = WaitingTimePredictor()


# =========================================================
# Helper Functions
# =========================================================

def get_queue_count(service_id):

    return QueueToken.query.filter_by(

        service_id=service_id,

        status="waiting"

    ).count()


def get_current_token(service_id):

    return QueueToken.query.filter(

        QueueToken.service_id == service_id,

        QueueToken.status == "serving"

    ).first()


def generate_token_number(service_id):

    last_token = QueueToken.query.filter_by(

        service_id=service_id

    ).order_by(

        QueueToken.token_number.desc()

    ).first()

    if last_token:

        return last_token.token_number + 1

    return 1


# =========================================================
# Home Page
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# API: Get Services
# =========================================================

@app.route(
    "/api/services",
    methods=["GET"]
)
def get_services():

    services = Service.query.filter_by(
        active=True
    ).all()

    result = []

    for service in services:

        queue_count = get_queue_count(
            service.id
        )

        estimated_time = predictor.predict(

            queue_count,

            service.avg_service_time,

            1

        )

        result.append({

            "id":
                service.id,

            "name":
                service.name,

            "queue_count":
                queue_count,

            "average_service_time":
                service.avg_service_time,

            "estimated_waiting_time":
                estimated_time
        })

    return jsonify(result)


# =========================================================
# API: Join Queue
# =========================================================

@app.route(
    "/api/join-queue",
    methods=["POST"]
)
def join_queue():

    data = request.get_json()

    if not data:

        return jsonify({

            "error":
                "Request body is required"

        }), 400

    student_name = str(
        data.get("student_name", "")
    ).strip()

    student_id = str(
        data.get("student_id", "")
    ).strip()

    service_id = data.get(
        "service_id"
    )

    # -----------------------------
    # Validate Student Name
    # -----------------------------

    if not student_name:

        return jsonify({

            "error":
                "Student name is required"

        }), 400

    # -----------------------------
    # Validate Student ID
    # -----------------------------

    if not student_id:

        return jsonify({

            "error":
                "Student ID is required"

        }), 400

    # -----------------------------
    # Validate Service
    # -----------------------------

    if service_id is None:

        return jsonify({

            "error":
                "Service ID is required"

        }), 400

    try:

        service_id = int(service_id)

    except (TypeError, ValueError):

        return jsonify({

            "error":
                "Invalid service ID"

        }), 400

    service = db.session.get(
        Service,
        service_id
    )

    if not service:

        return jsonify({

            "error":
                "Service not found"

        }), 404

    # -----------------------------
    # Check Existing Token
    # -----------------------------

    existing = QueueToken.query.filter(

        QueueToken.student_id == student_id,

        QueueToken.service_id == service_id,

        QueueToken.status.in_([
            "waiting",
            "serving"
        ])

    ).first()

    if existing:

        return jsonify({

            "error":
                "You already have an active token",

            "token_id":
                existing.id,

            "token":
                existing.token_number

        }), 400

    # -----------------------------
    # Queue Information
    # -----------------------------

    people_waiting = get_queue_count(
        service_id
    )

    token_number = generate_token_number(
        service_id
    )

    # -----------------------------
    # Create Token
    # -----------------------------

    token = QueueToken(

        token_number=token_number,

        student_name=student_name,

        student_id=student_id,

        service_id=service_id,

        status="waiting"

    )

    db.session.add(token)

    db.session.commit()

    # -----------------------------
    # AI Prediction
    # -----------------------------

    estimated_time = predictor.predict(

        people_waiting + 1,

        service.avg_service_time,

        1

    )

    position = people_waiting + 1

    return jsonify({

        "message":
            "Successfully joined queue",

        "token_id":
            token.id,

        "token":
            token.token_number,

        "position":
            position,

        "estimated_waiting_time":
            estimated_time,

        "service":
            service.name
    }), 201


# =========================================================
# API: Token Status
# =========================================================

@app.route(
    "/api/token/<int:token_id>",
    methods=["GET"]
)
def token_status(token_id):

    token = db.session.get(
        QueueToken,
        token_id
    )

    if not token:

        return jsonify({

            "error":
                "Token not found"

        }), 404

    people_ahead = QueueToken.query.filter(

        QueueToken.service_id ==
        token.service_id,

        QueueToken.status == "waiting",

        QueueToken.created_at <
        token.created_at

    ).count()

    service = token.service

    estimated_time = predictor.predict(

        people_ahead,

        service.avg_service_time,

        1

    )

    return jsonify({

        "token_id":
            token.id,

        "token_number":
            token.token_number,

        "student_name":
            token.student_name,

        "student_id":
            token.student_id,

        "service":
            service.name,

        "status":
            token.status,

        "people_ahead":
            people_ahead,

        "position":
            people_ahead + 1,

        "estimated_waiting_time":
            estimated_time,

        "created_at":
            token.created_at.isoformat()
    })


# =========================================================
# API: Get Current Queue
# =========================================================

@app.route(
    "/api/queue/<int:service_id>",
    methods=["GET"]
)
def get_queue(service_id):

    service = db.session.get(
        Service,
        service_id
    )

    if not service:

        return jsonify({

            "error":
                "Service not found"

        }), 404

    current = get_current_token(
        service_id
    )

    waiting_tokens = QueueToken.query.filter_by(

        service_id=service_id,

        status="waiting"

    ).order_by(

        QueueToken.created_at.asc()

    ).all()

    queue = []

    for token in waiting_tokens:

        queue.append({

            "id":
                token.id,

            "token":
                token.token_number,

            "student_name":
                token.student_name,

            "status":
                token.status
        })

    return jsonify({

        "service":
            service.name,

        "current_token":
            current.token_number
            if current else None,

        "current_token_id":
            current.id
            if current else None,

        "waiting_count":
            len(waiting_tokens),

        "queue":
            queue
    })


# =========================================================
# API: Call Next Student
# =========================================================

@app.route(
    "/api/call-next/<int:service_id>",
    methods=["POST"]
)
def call_next(service_id):

    service = db.session.get(
        Service,
        service_id
    )

    if not service:

        return jsonify({

            "error":
                "Service not found"

        }), 404

    current = get_current_token(
        service_id
    )

    if current:

        return jsonify({

            "error":
                f"Token {current.token_number} "
                "is currently being served"

        }), 400

    next_token = QueueToken.query.filter_by(

        service_id=service_id,

        status="waiting"

    ).order_by(

        QueueToken.created_at.asc()

    ).first()

    if not next_token:

        return jsonify({

            "message":
                "No students waiting"

        }), 404

    next_token.status = "serving"

    next_token.called_at = datetime.now(
        timezone.utc
    )

    db.session.commit()

    return jsonify({

        "message":
            "Next student called",

        "token_id":
            next_token.id,

        "token":
            next_token.token_number,

        "student_name":
            next_token.student_name,

        "student_id":
            next_token.student_id,

        "service":
            service.name
    })


# =========================================================
# API: Complete Current Service
# =========================================================

@app.route(
    "/api/complete/<int:service_id>",
    methods=["POST"]
)
def complete_service(service_id):

    current = get_current_token(
        service_id
    )

    if not current:

        return jsonify({

            "error":
                "No student is currently being served"

        }), 404

    current.status = "completed"

    current.completed_at = datetime.now(
        timezone.utc
    )

    db.session.commit()

    return jsonify({

        "message":
            "Service completed",

        "token_id":
            current.id,

        "token":
            current.token_number
    })


# =========================================================
# API: Cancel Token
# =========================================================

@app.route(
    "/api/cancel/<int:token_id>",
    methods=["POST"]
)
def cancel_token(token_id):

    token = db.session.get(
        QueueToken,
        token_id
    )

    if not token:

        return jsonify({

            "error":
                "Token not found"

        }), 404

    if token.status != "waiting":

        return jsonify({

            "error":
                "Only waiting tokens can be cancelled"

        }), 400

    token.status = "cancelled"

    db.session.commit()

    return jsonify({

        "message":
            "Token cancelled successfully",

        "token":
            token.token_number
    })


# =========================================================
# API: Dashboard Statistics
# =========================================================

@app.route(
    "/api/dashboard",
    methods=["GET"]
)
def dashboard():

    total_tokens = QueueToken.query.count()

    waiting = QueueToken.query.filter_by(
        status="waiting"
    ).count()

    serving = QueueToken.query.filter_by(
        status="serving"
    ).count()

    completed = QueueToken.query.filter_by(
        status="completed"
    ).count()

    cancelled = QueueToken.query.filter_by(
        status="cancelled"
    ).count()

    return jsonify({

        "total_tokens":
            total_tokens,

        "waiting":
            waiting,

        "serving":
            serving,

        "completed":
            completed,

        "cancelled":
            cancelled
    })


# =========================================================
# Create Default Campus Services
# =========================================================

def create_default_services():

    default_services = [

        {
            "name": "Canteen",
            "avg_service_time": 4
        },

        {
            "name": "Library",
            "avg_service_time": 5
        },

        {
            "name": "Student Office",
            "avg_service_time": 8
        },

        {
            "name": "Fee Counter",
            "avg_service_time": 10
        },

        {
            "name": "Certificates",
            "avg_service_time": 7
        },

        {
            "name": "Lab Equipment",
            "avg_service_time": 6
        }

    ]

    for item in default_services:

        existing = Service.query.filter_by(
            name=item["name"]
        ).first()

        if not existing:

            service = Service(

                name=item["name"],

                avg_service_time=
                    item["avg_service_time"],

                active=True

            )

            db.session.add(service)

    db.session.commit()


# =========================================================
# Database Initialization
# =========================================================

with app.app_context():

    db.create_all()

    create_default_services()


# =========================================================
# Run Application
# =========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )
