"""
Evaluation dataset for the Potens RAG Document Q&A project.

Used for:
- End-to-end RAG evaluation
- Retrieval evaluation
- Citation evaluation
- RAGAS evaluation
- Hallucination / abstention testing
"""


# ============================================================
# DOCUMENT SOURCE NAMES
# ============================================================
# IMPORTANT:
# These names should match doc.metadata["source"] in ChromaDB.
# Change them if your actual filenames are different.

EMPLOYEE_HANDBOOK = "employee_handbook.pdf"
LEAVE_POLICY = "leave_policy.pdf"
REMOTE_WORK_POLICY = "remote_work_policy.pdf"
SECURITY_POLICY = "security_policy.pdf"
TRAVEL_POLICY = "travel_policy.pdf"


# ============================================================
# EVALUATION DATASET
# ============================================================

EVAL_DATASET = [

    # ========================================================
    # 1. EMPLOYEE HANDBOOK
    # ========================================================

    {
        "id": "HB001",
        "question": "What is the probation period for new employees?",
        "expected_answer": (
            "New employees serve a probation period of 90 days "
            "from their date of joining."
        ),
        "expected_sources": [EMPLOYEE_HANDBOOK],
        "category": "factual",
    },

    {
        "id": "HB002",
        "question": "What are the standard working hours at Acme Corporation?",
        "expected_answer": (
            "Standard working hours are 9:30 AM to 6:30 PM, "
            "Monday through Friday, with a one-hour lunch break."
        ),
        "expected_sources": [EMPLOYEE_HANDBOOK],
        "category": "factual",
    },

    {
        "id": "HB003",
        "question": "What are the core values of Acme Corporation?",
        "expected_answer": (
            "Acme Corporation's core values are Integrity, "
            "Teamwork, Excellence, and Respect."
        ),
        "expected_sources": [EMPLOYEE_HANDBOOK],
        "category": "factual",
    },

    {
        "id": "HB004",
        "question": "When are salaries paid to employees?",
        "expected_answer": (
            "Salaries are credited to employees' bank accounts "
            "on the last working day of every month."
        ),
        "expected_sources": [EMPLOYEE_HANDBOOK],
        "category": "factual",
    },

    {
        "id": "HB005",
        "question": "How much certification reimbursement can an employee claim per year?",
        "expected_answer": (
            "Employees can claim reimbursement for job-relevant "
            "certifications up to Rs. 25,000 per year after approval "
            "from their reporting manager."
        ),
        "expected_sources": [EMPLOYEE_HANDBOOK],
        "category": "factual",
    },

    {
        "id": "HB006",
        "question": "When are formal performance reviews conducted?",
        "expected_answer": (
            "Formal performance reviews are conducted twice a year, "
            "in June and December."
        ),
        "expected_sources": [EMPLOYEE_HANDBOOK],
        "category": "factual",
    },

    {
        "id": "HB007",
        "question": "How long does the full and final settlement process take?",
        "expected_answer": (
            "The full and final settlement is processed within "
            "45 days of the employee's last working day."
        ),
        "expected_sources": [EMPLOYEE_HANDBOOK],
        "category": "factual",
    },

    # ========================================================
    # 2. LEAVE POLICY
    # ========================================================

    {
        "id": "LV001",
        "question": "How many casual leave days are employees entitled to each year?",
        "expected_answer": (
            "Employees are entitled to 12 days of casual leave "
            "per calendar year, credited at the rate of 1 day per month."
        ),
        "expected_sources": [LEAVE_POLICY],
        "category": "factual",
    },

    {
        "id": "LV002",
        "question": "How many sick leave days are available per year?",
        "expected_answer": (
            "Employees are entitled to 10 days of sick leave "
            "per calendar year."
        ),
        "expected_sources": [LEAVE_POLICY],
        "category": "factual",
    },

    {
        "id": "LV003",
        "question": "When is a medical certificate required for sick leave?",
        "expected_answer": (
            "A medical certificate is required for sick leave "
            "exceeding 2 consecutive days."
        ),
        "expected_sources": [LEAVE_POLICY],
        "category": "factual",
    },

    {
        "id": "LV004",
        "question": "How much earned leave can an employee carry forward?",
        "expected_answer": (
            "Earned leave can be carried forward up to a maximum "
            "of 45 days."
        ),
        "expected_sources": [LEAVE_POLICY],
        "category": "factual",
    },

    {
        "id": "LV005",
        "question": "How much paid maternity leave is provided?",
        "expected_answer": (
            "Female employees are entitled to 26 weeks of paid "
            "maternity leave as per applicable statutory regulations, "
            "for up to two children."
        ),
        "expected_sources": [LEAVE_POLICY],
        "category": "factual",
    },

    {
        "id": "LV006",
        "question": "How much paid paternity leave is available?",
        "expected_answer": (
            "Male employees are entitled to 2 weeks of paid paternity "
            "leave, to be availed within 3 months of the child's birth."
        ),
        "expected_sources": [LEAVE_POLICY],
        "category": "factual",
    },

    {
        "id": "LV007",
        "question": "How far in advance should an employee apply for leave?",
        "expected_answer": (
            "Leave requests must be submitted through the HR portal "
            "at least 2 working days in advance, except in cases of "
            "emergency or illness."
        ),
        "expected_sources": [LEAVE_POLICY],
        "category": "factual",
    },

    {
        "id": "LV008",
        "question": "What happens to unused casual leave and sick leave?",
        "expected_answer": (
            "Unused casual leave and sick leave cannot be encashed "
            "and will lapse at the end of the calendar year."
        ),
        "expected_sources": [LEAVE_POLICY],
        "category": "factual",
    },

    # ========================================================
    # 3. REMOTE WORK POLICY
    # ========================================================

    {
        "id": "RW001",
        "question": "What factors determine whether an employee is eligible for remote work?",
        "expected_answer": (
            "Remote work eligibility depends on the nature of the role, "
            "business requirements, and the employee's performance record."
        ),
        "expected_sources": [REMOTE_WORK_POLICY],
        "category": "factual",
    },

    {
        "id": "RW002",
        "question": "How many days per week do hybrid employees typically work from the office?",
        "expected_answer": (
            "Hybrid employees typically work from the office "
            "2 to 3 days per week."
        ),
        "expected_sources": [REMOTE_WORK_POLICY],
        "category": "factual",
    },

    {
        "id": "RW003",
        "question": "What are the standard working hours for remote employees?",
        "expected_answer": (
            "Remote employees are expected to be available and responsive "
            "during standard business hours from 9:30 AM to 6:30 PM, "
            "unless otherwise agreed with their manager."
        ),
        "expected_sources": [REMOTE_WORK_POLICY],
        "category": "factual",
    },

    {
        "id": "RW004",
        "question": "What equipment does Acme Corporation provide to remote employees?",
        "expected_answer": (
            "Acme Corporation provides a laptop and, where required, "
            "additional equipment such as a monitor, keyboard, and headset."
        ),
        "expected_sources": [REMOTE_WORK_POLICY],
        "category": "factual",
    },

    {
        "id": "RW005",
        "question": "How much internet and utility reimbursement can remote employees receive?",
        "expected_answer": (
            "Remote employees are eligible for a monthly internet "
            "and utility reimbursement of up to Rs. 1,500, subject "
            "to submission of valid bills."
        ),
        "expected_sources": [REMOTE_WORK_POLICY],
        "category": "factual",
    },

    {
        "id": "RW006",
        "question": "Can employees use public Wi-Fi to access company systems while working remotely?",
        "expected_answer": (
            "Public or unsecured Wi-Fi networks must not be used to "
            "access company systems unless the employee is connected "
            "through the company VPN."
        ),
        "expected_sources": [REMOTE_WORK_POLICY],
        "category": "factual",
    },

    # ========================================================
    # 4. SECURITY POLICY
    # ========================================================

    {
        "id": "SEC001",
        "question": "What are the four information classification categories used by Acme Corporation?",
        "expected_answer": (
            "The four information classification categories are "
            "Public, Internal, Confidential, and Restricted."
        ),
        "expected_sources": [SECURITY_POLICY],
        "category": "factual",
    },

    {
        "id": "SEC002",
        "question": "What are the company's password requirements?",
        "expected_answer": (
            "Passwords must be at least 12 characters long, include "
            "uppercase and lowercase letters, numbers, and special "
            "characters, and must be changed every 90 days."
        ),
        "expected_sources": [SECURITY_POLICY],
        "category": "factual",
    },

    {
        "id": "SEC003",
        "question": "For which systems is multi-factor authentication mandatory?",
        "expected_answer": (
            "Multi-factor authentication is mandatory for accessing "
            "company email, VPN, and all critical business applications."
        ),
        "expected_sources": [SECURITY_POLICY],
        "category": "factual",
    },

    {
        "id": "SEC004",
        "question": "Can employees share their login credentials with IT support staff?",
        "expected_answer": (
            "No. Employees must never share their login credentials "
            "with anyone, including IT support staff or colleagues."
        ),
        "expected_sources": [SECURITY_POLICY],
        "category": "factual",
    },

    {
        "id": "SEC005",
        "question": "What should an employee do after discovering a suspected security incident?",
        "expected_answer": (
            "The employee must report the incident to the IT Security "
            "team immediately, using security@acmecorp.com or the IT "
            "helpdesk within 1 hour of discovery. The employee should "
            "not investigate or resolve the incident independently and "
            "should preserve relevant evidence."
        ),
        "expected_sources": [SECURITY_POLICY],
        "category": "factual",
    },

    {
        "id": "SEC006",
        "question": "Can employees install unauthorized software on company devices?",
        "expected_answer": (
            "No. Employees must not install unauthorized or unlicensed "
            "software on company devices. Software requests must go "
            "through the IT helpdesk."
        ),
        "expected_sources": [SECURITY_POLICY],
        "category": "factual",
    },

    # ========================================================
    # 5. TRAVEL POLICY
    # ========================================================

    {
        "id": "TR001",
        "question": "How far in advance should domestic business travel be requested?",
        "expected_answer": (
            "Domestic travel requests should be submitted at least "
            "5 working days in advance."
        ),
        "expected_sources": [TRAVEL_POLICY],
        "category": "factual",
    },

    {
        "id": "TR002",
        "question": "How far in advance should international business travel be requested?",
        "expected_answer": (
            "International travel requests require a minimum of "
            "15 working days for visa processing."
        ),
        "expected_sources": [TRAVEL_POLICY],
        "category": "factual",
    },

    {
        "id": "TR003",
        "question": "What class of flight can managers and below use for domestic travel?",
        "expected_answer": (
            "Managers and below are entitled to economy class "
            "for all domestic flights."
        ),
        "expected_sources": [TRAVEL_POLICY],
        "category": "factual",
    },

    {
        "id": "TR004",
        "question": "Within how many days must employees submit travel expense claims?",
        "expected_answer": (
            "Travel-related expenses must be submitted through the "
            "Expense Management System within 15 days of completing "
            "the trip, along with original receipts."
        ),
        "expected_sources": [TRAVEL_POLICY],
        "category": "factual",
    },

    {
        "id": "TR005",
        "question": "Which travel expenses are not reimbursable?",
        "expected_answer": (
            "Non-reimbursable expenses include personal entertainment "
            "and mini-bar charges, traffic fines and penalties, expenses "
            "for family members accompanying the employee, and alcohol "
            "and tobacco purchases."
        ),
        "expected_sources": [TRAVEL_POLICY],
        "category": "factual",
    },

    # ========================================================
    # 6. MULTI-DOCUMENT QUESTIONS
    # ========================================================

    {
        "id": "MULTI001",
        "question": (
            "If I am working remotely, what are my normal working "
            "hours and what security measure should I use when "
            "accessing internal company systems?"
        ),
        "expected_answer": (
            "Remote employees are normally expected to be available "
            "from 9:30 AM to 6:30 PM. When accessing internal systems "
            "or confidential data from a remote location, employees "
            "must connect through the company VPN."
        ),
        "expected_sources": [
            REMOTE_WORK_POLICY,
        ],
        "category": "multi_document",
    },

    {
        "id": "MULTI002",
        "question": (
            "I am working remotely and need to take three consecutive "
            "days of sick leave. What documentation is required, and "
            "what security requirement applies when accessing internal "
            "systems remotely?"
        ),
        "expected_answer": (
            "A medical certificate is required because the sick leave "
            "exceeds 2 consecutive days. When accessing internal systems "
            "or confidential data remotely, the employee must use the "
            "company VPN."
        ),
        "expected_sources": [
            LEAVE_POLICY,
            REMOTE_WORK_POLICY,
        ],
        "category": "multi_document",
    },

    {
        "id": "MULTI003",
        "question": (
            "An employee is traveling on official business and discovers "
            "a phishing email. What travel protection applies and what "
            "should the employee do about the security incident?"
        ),
        "expected_answer": (
            "Employees traveling on official business are covered by "
            "the company's corporate travel insurance for the duration "
            "of the trip. The suspected phishing incident must be "
            "reported to the IT Security team immediately, through "
            "security@acmecorp.com or the IT helpdesk within 1 hour "
            "of discovery."
        ),
        "expected_sources": [
            TRAVEL_POLICY,
            SECURITY_POLICY,
        ],
        "category": "multi_document",
    },

    {
        "id": "MULTI004",
        "question": (
            "If an employee resigns with unused earned leave, what does "
            "the handbook say about final settlement and what does the "
            "leave policy say about earned leave carry forward?"
        ),
        "expected_answer": (
            "The full and final settlement, including leave encashment "
            "and other dues, is processed within 45 days of the last "
            "working day. Earned leave can be carried forward up to "
            "a maximum of 45 days."
        ),
        "expected_sources": [
            EMPLOYEE_HANDBOOK,
            LEAVE_POLICY,
        ],
        "category": "multi_document",
    },

    # ========================================================
    # 7. UNANSWERABLE / HALLUCINATION TESTS
    # ========================================================

    {
        "id": "NEG001",
        "question": "What is the CEO's annual salary?",
        "expected_answer": (
            "This information is not provided in the available documents."
        ),
        "expected_sources": [],
        "category": "unanswerable",
    },

    {
        "id": "NEG002",
        "question": "What is the Wi-Fi password of the Pune office?",
        "expected_answer": (
            "This information is not provided in the available documents."
        ),
        "expected_sources": [],
        "category": "unanswerable",
    },

    {
        "id": "NEG003",
        "question": "Which programming language does Acme Corporation use for its backend systems?",
        "expected_answer": (
            "This information is not provided in the available documents."
        ),
        "expected_sources": [],
        "category": "unanswerable",
    },

    # ========================================================
    # 8. PARTIALLY ANSWERABLE QUESTIONS
    # ========================================================

    {
        "id": "PART001",
        "question": "What is the exact hotel reimbursement limit per night for Mumbai?",
        "expected_answer": (
            "The exact reimbursement amount is not provided in the "
            "available documents. The Travel Policy states that hotel "
            "limits depend on the employee's city category and grade."
        ),
        "expected_sources": [TRAVEL_POLICY],
        "category": "partial_answer",
    },

    {
        "id": "PART002",
        "question": "What is the exact daily allowance amount for employees traveling on business?",
        "expected_answer": (
            "The exact daily allowance amount is not provided in the "
            "available documents. The Travel Policy states that per diem "
            "rates are published on the HR portal."
        ),
        "expected_sources": [TRAVEL_POLICY],
        "category": "partial_answer",
    },

    # ========================================================
    # 9. MULTILINGUAL TESTS
    # ========================================================

    {
        "id": "HI001",
        "question": "कर्मचारियों को साल में कितनी casual leave मिलती है?",
        "expected_answer": (
            "Employees are entitled to 12 days of casual leave "
            "per calendar year."
        ),
        "expected_sources": [LEAVE_POLICY],
        "category": "multilingual",
    },

    {
        "id": "HI002",
        "question": "Remote employee ko company VPN kab use karna chahiye?",
        "expected_answer": (
            "Remote employees must use the company VPN whenever "
            "accessing internal systems or confidential data from "
            "a remote location."
        ),
        "expected_sources": [REMOTE_WORK_POLICY],
        "category": "multilingual",
    },

    {
        "id": "MR001",
        "question": "कर्मचाऱ्यांसाठी probation period किती दिवसांचा आहे?",
        "expected_answer": (
            "The probation period for new employees is 90 days "
            "from the date of joining."
        ),
        "expected_sources": [EMPLOYEE_HANDBOOK],
        "category": "multilingual",
    },

    {
        "id": "MR002",
        "question": "Security incident सापडल्यानंतर किती वेळात report करणे आवश्यक आहे?",
        "expected_answer": (
            "A suspected security incident must be reported to the "
            "IT Security team immediately and through the designated "
            "email or IT helpdesk within 1 hour of discovery."
        ),
        "expected_sources": [SECURITY_POLICY],
        "category": "multilingual",
    },
]


# ============================================================
# OPTIONAL DATASET INFORMATION
# ============================================================

DATASET_INFO = {
    "name": "Potens RAG Evaluation Dataset",
    "version": "1.0",
    "total_questions": len(EVAL_DATASET),
    "documents": 5,
}