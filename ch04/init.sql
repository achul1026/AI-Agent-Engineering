-- ============================================================
-- 민간임대주택 관리 시스템 — 통합 DB 스키마
-- Version  : v2.0
-- DBMS     : PostgreSQL 14+
-- Encoding : UTF-8 / ko_KR.UTF-8
-- Created  : 2026-03-26
-- ============================================================

-- ── 확장 모듈 ────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pgcrypto;      -- 개인정보 암호화 지원
CREATE EXTENSION IF NOT EXISTS pgvector;      -- AI RAG 벡터 저장 (3단계)

-- ============================================================
-- SECTION 1. 공통 인프라 (RBAC · 조직 · 사용자)
-- ============================================================

-- 1-1. 역할 (Role) 마스터
--   SYSTEM_ADMIN · EXECUTIVE · OPS_MANAGER · FIELD_STAFF
--   REPORT_STAFF · TENANT · AI_AGENT
CREATE TABLE roles (
                       id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                       code            VARCHAR(30)  NOT NULL UNIQUE,   -- SYSTEM_ADMIN, OPS_MANAGER …
                       name            VARCHAR(50)  NOT NULL,           -- 역할 한글명
                       description     TEXT,
                       created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                       updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 1-2. 조직 (Organization)
--   HEAD_OFFICE(본사) · ASSOCIATION(조합) · SPC(SPC 법인)
--   AS_PARTNER(협력업체) · REAL_ESTATE(공인중개사)
CREATE TABLE organizations (
                               id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                               org_type        VARCHAR(30)  NOT NULL,           -- HEAD_OFFICE | ASSOCIATION | SPC | AS_PARTNER | REAL_ESTATE
                               parent_id       UUID REFERENCES organizations(id), -- SPC 계층 구조 (미확정 시 NULL)
                               name            VARCHAR(100) NOT NULL,
                               biz_reg_no      VARCHAR(20)  UNIQUE,             -- 사업자등록번호 (암호화)
                               representative  VARCHAR(50),                     -- 대표자명
                               contact_enc     VARCHAR(200),                    -- 대표 연락처 (AES-256)
                               address         TEXT,
                               bank_name       VARCHAR(50),
                               account_enc     VARCHAR(200),                    -- 계좌번호 (AES-256)
                               seal_image_path TEXT,                            -- 법인 인감 이미지 경로 (공문 자동생성용)
                               notify_kakao    VARCHAR(50),                     -- 카카오 알림 수신 번호
                               notify_opt_out  BOOLEAN      NOT NULL DEFAULT FALSE,
                               is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
                               created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                               updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

COMMENT ON COLUMN organizations.parent_id IS 'SPC 법인 계층 구조 — 미확정 시 NULL 유지';

-- 1-3. 시스템 사용자 (내부 직원 전용)
CREATE TABLE users (
                       id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                       login_id        VARCHAR(50)  NOT NULL UNIQUE,
                       password_hash   TEXT         NOT NULL,           -- bcrypt (cost 12)
                       name            VARCHAR(50)  NOT NULL,
                       email           VARCHAR(100) UNIQUE,
                       phone_enc       VARCHAR(200),                    -- 연락처 (AES-256)
                       role_id         UUID         NOT NULL REFERENCES roles(id),
                       is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
                       last_login_at   TIMESTAMPTZ,
                       created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                       updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 1-4. 사용자 ↔ 현장 다대다 매핑 (Row-Level Security 기반)
CREATE TABLE user_site_mappings (
                                    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                                    site_id         UUID NOT NULL,                   -- FK → rental_sites.id (후술)
                                    granted_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                                    granted_by      UUID REFERENCES users(id),
                                    PRIMARY KEY (user_id, site_id)
);

-- ============================================================
-- SECTION 2. 자산 관리 (단지 · 주택 · 소유권)
-- ============================================================

-- 2-1. 임대단지 (Rental Site)
CREATE TABLE rental_sites (
                              id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                              owner_org_id        UUID         NOT NULL REFERENCES organizations(id), -- 소유 조직
                              managing_org_id     UUID         REFERENCES organizations(id),          -- 관리 위탁 조직 (본사)
                              name                VARCHAR(100) NOT NULL,
                              address             TEXT         NOT NULL,
                              region_code         VARCHAR(10)  NOT NULL,           -- 시도·시군구 코드
                              rental_type_code    VARCHAR(20)  NOT NULL,           -- 공통코드 참조 (민간일반·공공지원 등)
                              total_units         INTEGER      NOT NULL DEFAULT 0,
                              status              VARCHAR(20)  NOT NULL DEFAULT 'OPERATING', -- OPERATING | PRE_COMPLETION | CLOSED
                              memo                TEXT,
                              created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                              updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 2-2. 개별 임대주택 (Rental Unit)
CREATE TABLE rental_units (
                              id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                              site_id                 UUID         NOT NULL REFERENCES rental_sites(id),
                              unit_no                 VARCHAR(20)  NOT NULL,       -- 호실 번호
                              building_no             VARCHAR(20),                 -- 동 번호
                              floor                   SMALLINT,
                              unit_type_code          VARCHAR(20)  NOT NULL,       -- 공통코드 (APT·OFFICETEL·URBAN_HOUSING)
                              exclusive_area_m2       NUMERIC(8,2) NOT NULL,       -- 전용면적 (㎡)
                              supply_area_m2          NUMERIC(8,2),                -- 공급면적 (㎡)
                              status                  VARCHAR(20)  NOT NULL DEFAULT 'VACANT', -- VACANT | OCCUPIED | MAINTENANCE | HOLD
                              standard_deposit        BIGINT,                      -- 기준 보증금 (원)
                              standard_rent           INTEGER,                     -- 기준 월임대료 (원)
                              sub_registration_date   DATE,                        -- 부기등기일
                              is_sub_registered       BOOLEAN      NOT NULL DEFAULT FALSE, -- 부기등기 완료 여부
                              created_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                              updated_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                              UNIQUE (site_id, building_no, unit_no)
);

COMMENT ON COLUMN rental_units.sub_registration_date IS '민간임대주택법 의무 부기등기 관리';
COMMENT ON COLUMN rental_units.is_sub_registered    IS '부기등기 완료 여부';

-- 2-3. 소유권 이전 · 양도 이력 (포괄양수도 / 매각 등)
CREATE TABLE ownership_transfer_logs (
                                         id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                                         unit_id         UUID         NOT NULL REFERENCES rental_units(id),
                                         from_org_id     UUID         REFERENCES organizations(id),
                                         to_org_id       UUID         NOT NULL REFERENCES organizations(id),
                                         transfer_date   DATE         NOT NULL,
                                         transfer_type   VARCHAR(30)  NOT NULL,   -- COMPREHENSIVE_TRANSFER | SALE | INHERITANCE
                                         reason          TEXT,
                                         doc_path        TEXT,                    -- 증빙 서류 경로
                                         created_by      UUID         REFERENCES users(id),
                                         created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ============================================================
-- SECTION 3. 임차인 · 계약 (Tenant · Contract)
-- ============================================================

-- 3-1. 임차인
CREATE TABLE tenants (
                         id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                         name                VARCHAR(50)  NOT NULL,
                         resident_no_enc     VARCHAR(256) NOT NULL,       -- 주민등록번호 (AES-256)
                         phone_enc           VARCHAR(200) NOT NULL,        -- 휴대폰 번호 (AES-256)
                         email               VARCHAR(100),
                         bank_code           VARCHAR(10),
                         account_enc         VARCHAR(256),                -- 계좌번호 (AES-256)
                         portal_account      VARCHAR(100) UNIQUE,         -- 임차인 포털 로그인 계정
                         privacy_agreed_at   TIMESTAMPTZ,                 -- 개인정보 수집·이용 동의 일시
                         notify_opt_out      BOOLEAN      NOT NULL DEFAULT FALSE,
                         created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                         updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 3-2. 임차인 연락처 변경 이력
CREATE TABLE tenant_contact_histories (
                                          id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                                          tenant_id       UUID         NOT NULL REFERENCES tenants(id),
                                          phone_enc       VARCHAR(200) NOT NULL,
                                          changed_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                                          changed_by      UUID         REFERENCES users(id)
);

-- 3-3. 임대차 계약 (핵심 테이블)
CREATE TABLE lease_contracts (
                                 id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                                 unit_id             UUID         NOT NULL REFERENCES rental_units(id),
                                 tenant_id           UUID         NOT NULL REFERENCES tenants(id),
                                 owner_org_id        UUID         NOT NULL REFERENCES organizations(id), -- 임대사업자
                                 created_by          UUID         NOT NULL REFERENCES users(id),
                                 contract_no         VARCHAR(30)  NOT NULL UNIQUE,   -- 자동채번
                                 contract_type_code  VARCHAR(20)  NOT NULL,           -- CT_NEW | CT_RENEWAL
                                 start_date          DATE         NOT NULL,
                                 end_date            DATE         NOT NULL,
                                 signed_date         DATE,                            -- 계약 체결일
                                 report_deadline     DATE,                            -- 신고 기한 (체결일 + 30일)
                                 deposit             BIGINT       NOT NULL,
                                 monthly_rent        INTEGER      NOT NULL DEFAULT 0,
                                 rent_due_day        SMALLINT     NOT NULL,           -- 월 납부일 (1~28)
                                 rent_increase_rate  NUMERIC(5,2) NOT NULL DEFAULT 5.0, -- 임대료 인상 제한율 (법정 5%)
                                 balance_amount      BIGINT,                          -- 잔금
                                 balance_due_date    DATE,                            -- 잔금 납부 예정일
                                 status              VARCHAR(20)  NOT NULL DEFAULT 'DRAFT', -- DRAFT | ACTIVE | EXPIRED | TERMINATED
                                 sign_status         VARCHAR(20)  NOT NULL DEFAULT 'PENDING', -- PENDING | SIGNED | REJECTED
                                 sign_request_id     VARCHAR(100),                   -- 사인오케이 서명 요청 ID
                                 signed_at           TIMESTAMPTZ,                     -- 전자서명 완료 일시
                                 terminated_at       DATE,                            -- 중도 해지일
                                 special_terms       TEXT,                            -- 특약사항 (AI 분석 데이터 소스)
                                 memo                TEXT,
                                 created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                                 updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

COMMENT ON COLUMN lease_contracts.rent_increase_rate IS '민간임대주택법 임대료 5% 인상 제한';
COMMENT ON COLUMN lease_contracts.special_terms      IS 'AI 특약사항 분석용 원문 텍스트';
COMMENT ON COLUMN lease_contracts.report_deadline    IS '임대차 신고 기한 — 계약 체결일 + 30일';

-- 3-4. 계약 상태 변경 이력
CREATE TABLE lease_contract_histories (
                                          id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                                          contract_id     UUID         NOT NULL REFERENCES lease_contracts(id),
                                          prev_status     VARCHAR(20),
                                          next_status     VARCHAR(20)  NOT NULL,
                                          changed_by      UUID         REFERENCES users(id),
                                          reason          TEXT,
                                          changed_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 3-5. 전자서명 요청 이력 (사인오케이 연동)
CREATE TABLE esign_requests (
                                id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                                contract_id         UUID         NOT NULL REFERENCES lease_contracts(id),
                                signok_request_id   VARCHAR(100) NOT NULL UNIQUE,   -- 사인오케이 발급 요청 ID
                                sign_url            TEXT,                            -- 서명 URL (임차인 SMS 발송용)
                                request_type        VARCHAR(20)  NOT NULL DEFAULT 'CONTRACT', -- CONTRACT | MOVE_OUT
                                status              VARCHAR(20)  NOT NULL DEFAULT 'PENDING',  -- PENDING | SIGNED | REJECTED | EXPIRED
                                requested_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                                completed_at        TIMESTAMPTZ,
                                signok_payload      JSONB                            -- Webhook 수신 전체 응답 원문
);

-- ============================================================
-- SECTION 4. 보증 · 수납 (Guarantee · Payment)
-- ============================================================

-- 4-1. 임대보증금 보증 (HUG · SGI · 은행)
CREATE TABLE deposit_guarantees (
                                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                                    contract_id     UUID         NOT NULL REFERENCES lease_contracts(id),
                                    agency          VARCHAR(50)  NOT NULL,   -- HUG | SGI | 은행명
                                    guarantee_no    VARCHAR(50)  UNIQUE,
                                    amount          BIGINT       NOT NULL,
                                    issued_date     DATE,
                                    expire_date     DATE         NOT NULL,   -- 만기일 (알림 기준)
                                    status          VARCHAR(20)  NOT NULL DEFAULT 'APPLIED', -- APPLIED | ISSUED | EXPIRED | CANCELLED
                                    doc_path        TEXT,                    -- 보증서 파일 경로
                                    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                                    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 4-2. 월별 임대료 수납
CREATE TABLE rent_payments (
                               id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                               contract_id         UUID         NOT NULL REFERENCES lease_contracts(id),
                               billing_year_month  CHAR(6)      NOT NULL,   -- YYYYMM
                               due_date            DATE         NOT NULL,
                               billed_amount       INTEGER      NOT NULL,
                               paid_date           DATE,                    -- NULL = 미납
                               paid_amount         INTEGER,
                               status              VARCHAR(20)  NOT NULL DEFAULT 'UNPAID', -- UNPAID | PAID | PARTIAL | OVERDUE
                               overdue_days        SMALLINT,                -- 연체 일수 (자동 계산)
                               note                VARCHAR(300),
                               created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                               UNIQUE (contract_id, billing_year_month)
);

-- ============================================================
-- SECTION 5. 신고 · 서식 (Report · Form)
-- ============================================================

-- 5-1. 신고 절차 관리
CREATE TABLE regulatory_reports (
                                    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                                    contract_id         UUID         NOT NULL REFERENCES lease_contracts(id),
                                    report_type_code    VARCHAR(20)  NOT NULL,   -- 공통코드 (REG_01 | CON_01 | TEL_01)
                                    status              VARCHAR(20)  NOT NULL DEFAULT 'PENDING', -- PENDING | IN_PROGRESS | SUBMITTED | CONFIRMED
                                    deadline            DATE,
                                    submitted_at        TIMESTAMPTZ,
                                    confirmed_at        TIMESTAMPTZ,
                                    handler_id          UUID         REFERENCES users(id),
                                    note                TEXT,
                                    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                                    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 5-2. 신고 서식 마스터 (버전 관리)
CREATE TABLE report_form_templates (
                                       id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                                       report_type_code VARCHAR(20) NOT NULL,
                                       version         VARCHAR(10)  NOT NULL,
                                       title           VARCHAR(200) NOT NULL,
                                       file_path       TEXT         NOT NULL,
                                       is_current      BOOLEAN      NOT NULL DEFAULT TRUE,
                                       effective_from  DATE         NOT NULL,
                                       created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ============================================================
-- SECTION 6. 민원 · 시설 (Complaint · Maintenance)
-- ============================================================

-- 6-1. 민원 · 하자보수
CREATE TABLE complaints (
                            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            unit_id         UUID         NOT NULL REFERENCES rental_units(id),
                            tenant_id       UUID         REFERENCES tenants(id),     -- 포털 접수 시 자동 연결
                            reported_by     UUID         REFERENCES users(id),       -- 내부 접수 시
                            category_code   VARCHAR(20)  NOT NULL,                   -- 공통코드 (설비·누수·입주전점검 등)
                            title           VARCHAR(200) NOT NULL,
                            description     TEXT         NOT NULL,
                            status          VARCHAR(20)  NOT NULL DEFAULT 'RECEIVED', -- RECEIVED | ASSIGNED | IN_PROGRESS | COMPLETED | DEFERRED
                            partner_org_id  UUID         REFERENCES organizations(id), -- 담당 AS 협력업체
                            assigned_at     TIMESTAMPTZ,
                            completed_at    TIMESTAMPTZ,
                            before_img_path TEXT,
                            after_img_path  TEXT,
                            repair_cost     NUMERIC(15,0) DEFAULT 0,
                            created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                            updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ============================================================
-- SECTION 7. 입주자 모집 · 중개 (Brokerage)
-- ============================================================

-- 7-1. 중개 수수료 정산
CREATE TABLE brokerage_fees (
                                id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                                contract_id     UUID         NOT NULL REFERENCES lease_contracts(id),
                                broker_org_id   UUID         NOT NULL REFERENCES organizations(id), -- 공인중개사 조직
                                fee_amount      NUMERIC(15,0) NOT NULL,
                                status          VARCHAR(20)  NOT NULL DEFAULT 'PENDING', -- PENDING | PAID | CANCELLED
                                paid_at         TIMESTAMPTZ,
                                note            TEXT,
                                created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ============================================================
-- SECTION 8. 알림 · 스케줄러 (Notification)
-- ============================================================

-- 8-1. 알림 발송 이력 (전 채널 통합)
CREATE TABLE notification_logs (
                                   id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                                   trigger_type    VARCHAR(50)  NOT NULL,   -- CONTRACT_EXPIRY | GUARANTEE_EXPIRY | UNPAID | SIGN_REQUEST | COMPLAINT_UPDATE …
                                   ref_table       VARCHAR(50)  NOT NULL,   -- 참조 테이블명
                                   ref_id          UUID         NOT NULL,   -- 참조 레코드 ID
                                   channel         VARCHAR(20)  NOT NULL,   -- KAKAO | SMS | EMAIL | SYSTEM
                                   recipient_type  VARCHAR(20)  NOT NULL,   -- TENANT | ORG | USER | EXTERNAL
                                   recipient_id    UUID,                    -- 내부 수신자 ID (user_id or org_id)
                                   recipient_contact_enc VARCHAR(200) NOT NULL, -- 발송 번호·이메일 (AES-256)
                                   message         TEXT         NOT NULL,
                                   status          VARCHAR(20)  NOT NULL DEFAULT 'PENDING', -- PENDING | SENT | FAILED | SKIPPED
                                   retry_count     SMALLINT     NOT NULL DEFAULT 0,
                                   scheduled_at    TIMESTAMPTZ,             -- 예약 발송 시각
                                   sent_at         TIMESTAMPTZ,
                                   error_message   TEXT,
                                   created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 8-2. 외부 수신자 등록 (법무사 · 회계사 · 세무사)
CREATE TABLE notification_recipients (
                                         id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                                         recipient_type  VARCHAR(30)  NOT NULL,   -- LEGAL_FIRM | ACCOUNTING | TAX
                                         name            VARCHAR(50)  NOT NULL,
                                         email           VARCHAR(100) NOT NULL,
                                         phone           VARCHAR(50),
                                         site_ids        UUID[],                  -- 담당 단지 목록
                                         is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
                                         created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ============================================================
-- SECTION 9. 문서 · 파일 (Document)
-- ============================================================

-- 9-1. 파일 첨부 이력
CREATE TABLE file_attachments (
                                  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                                  ref_table       VARCHAR(50)  NOT NULL,   -- lease_contracts | complaints | deposit_guarantees …
                                  ref_id          UUID         NOT NULL,
                                  file_type       VARCHAR(30)  NOT NULL,   -- CONTRACT_PDF | GUARANTEE | FORM | COMPLAINT_IMG | EXPORT
                                  file_path       TEXT         NOT NULL,
                                  original_name   VARCHAR(255),
                                  file_size_bytes INTEGER,
                                  uploaded_by     UUID         REFERENCES users(id),
                                  created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 9-2. 문서 아카이브 (AI 검색 + OCR)
CREATE TABLE document_archives (
                                   id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                                   ref_table       VARCHAR(50)  NOT NULL,   -- lease_contracts | LAW_GUIDE | OFFICIAL_DOC
                                   ref_id          UUID,
                                   title           VARCHAR(255) NOT NULL,
                                   doc_type        VARCHAR(30)  NOT NULL,   -- CONTRACT | LAW_GUIDE | OFFICIAL_DOC | REPORT_FORM
                                   ocr_content     TEXT,                    -- OCR 추출 텍스트 (AI 검색 소스)
                                   embedding       vector(1536),            -- pgvector 벡터 (AI RAG — 3단계)
                                   file_path       TEXT,
                                   created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

COMMENT ON COLUMN document_archives.embedding IS 'pgvector — M-14 AI Agent 3단계 구현 시 활성화';

-- ============================================================
-- SECTION 10. 공통코드 · 감사 (Common Code · Audit)
-- ============================================================

-- 10-1. 공통코드 마스터
CREATE TABLE common_codes (
                              id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                              group_code      VARCHAR(30)  NOT NULL,   -- CONTRACT_TYPE | UNIT_TYPE | REPORT_TYPE | COMPLAINT_CATEGORY …
                              code            VARCHAR(30)  NOT NULL,
                              name            VARCHAR(100) NOT NULL,
                              sort_order      SMALLINT     NOT NULL DEFAULT 0,
                              is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
                              created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                              UNIQUE (group_code, code)
);

-- 10-2. 감사 로그 (전체 CUD + 민감 VIEW 통합)
--   PK는 BIGSERIAL — 로그는 순차성이 중요
CREATE TABLE audit_logs (
                            id              BIGSERIAL    PRIMARY KEY,
                            user_id         UUID         REFERENCES users(id),   -- NULL = 시스템 자동
                            user_role       VARCHAR(30),                         -- 행위 시점 역할 스냅샷
                            action          VARCHAR(10)  NOT NULL,               -- INSERT | UPDATE | DELETE | VIEW
                            table_name      VARCHAR(50)  NOT NULL,
                            record_id       UUID,
                            before_data     JSONB,                               -- 변경 전 (UPDATE · DELETE)
                            after_data      JSONB,                               -- 변경 후 (INSERT · UPDATE)
                            ip_address      INET,
                            created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 10-3. 개인정보 접근 특화 로그 (법적 의무 별도 보관)
CREATE TABLE privacy_access_logs (
                                     id              BIGSERIAL    PRIMARY KEY,
                                     user_id         UUID         NOT NULL REFERENCES users(id),
                                     action_type     VARCHAR(50)  NOT NULL,   -- VIEW_RESIDENT_NO | UNMASK_ACCOUNT | PRINT_CONTRACT …
                                     target_tenant_id UUID        REFERENCES tenants(id),
                                     reason          TEXT,
                                     ip_address      INET,
                                     accessed_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE privacy_access_logs IS '개인정보보호법 제29조 접근 기록 의무 — 5년 보관';

-- ============================================================
-- SECTION 11. 인덱스
-- ============================================================

-- users
CREATE INDEX idx_users_role          ON users(role_id);
CREATE INDEX idx_users_login_id      ON users(login_id);

-- user_site_mappings
CREATE INDEX idx_usm_site            ON user_site_mappings(site_id);

-- rental_units
CREATE INDEX idx_units_site          ON rental_units(site_id);
CREATE INDEX idx_units_status        ON rental_units(status);

-- lease_contracts
CREATE INDEX idx_contracts_unit      ON lease_contracts(unit_id);
CREATE INDEX idx_contracts_tenant    ON lease_contracts(tenant_id);
CREATE INDEX idx_contracts_status    ON lease_contracts(status);
CREATE INDEX idx_contracts_end_date  ON lease_contracts(end_date);
CREATE INDEX idx_contracts_sign      ON lease_contracts(sign_status);

-- deposit_guarantees
CREATE INDEX idx_guarantees_contract ON deposit_guarantees(contract_id);
CREATE INDEX idx_guarantees_expire   ON deposit_guarantees(expire_date);
CREATE INDEX idx_guarantees_status   ON deposit_guarantees(status);

-- rent_payments
CREATE INDEX idx_payments_contract   ON rent_payments(contract_id);
CREATE INDEX idx_payments_ym         ON rent_payments(billing_year_month);
CREATE INDEX idx_payments_status     ON rent_payments(status);
CREATE INDEX idx_payments_due        ON rent_payments(due_date);

-- complaints
CREATE INDEX idx_complaints_unit     ON complaints(unit_id);
CREATE INDEX idx_complaints_status   ON complaints(status);

-- notification_logs
CREATE INDEX idx_notif_ref           ON notification_logs(ref_table, ref_id);
CREATE INDEX idx_notif_status        ON notification_logs(status);
CREATE INDEX idx_notif_created       ON notification_logs(created_at);

-- audit_logs
CREATE INDEX idx_audit_user          ON audit_logs(user_id);
CREATE INDEX idx_audit_table         ON audit_logs(table_name);
CREATE INDEX idx_audit_created       ON audit_logs(created_at);

-- document_archives (벡터 인덱스 — 3단계 시 활성화)
-- CREATE INDEX idx_doc_embedding ON document_archives USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ============================================================
-- SECTION 12. 공통코드 초기 데이터 (Seed)
-- ============================================================

INSERT INTO common_codes (group_code, code, name, sort_order) VALUES
-- 계약 유형
('CONTRACT_TYPE',   'CT_NEW',           '신규 계약',                    1),
('CONTRACT_TYPE',   'CT_RENEWAL',       '갱신 계약',                    2),
-- 주택 유형
('UNIT_TYPE',       'APT',              '아파트',                       1),
('UNIT_TYPE',       'OFFICETEL',        '오피스텔 (주거용)',              2),
('UNIT_TYPE',       'URBAN_HOUSING',    '도시형 생활주택',               3),
-- 호실 상태
('UNIT_STATUS',     'VACANT',           '공실',                         1),
('UNIT_STATUS',     'OCCUPIED',         '입주중',                       2),
('UNIT_STATUS',     'MAINTENANCE',      '관리중',                       3),
('UNIT_STATUS',     'HOLD',             '보류',                         4),
-- 계약 상태
('CONTRACT_STATUS', 'DRAFT',            '초안',                         1),
('CONTRACT_STATUS', 'ACTIVE',           '활성',                         2),
('CONTRACT_STATUS', 'EXPIRED',          '만기',                         3),
('CONTRACT_STATUS', 'TERMINATED',       '중도 해지',                    4),
-- 신고 유형
('REPORT_TYPE',     'REG_01',           '임대사업자 등록 신고 (최초)',    1),
('REPORT_TYPE',     'CON_01',           '임대차 계약 신고 (신규/변경/갱신)', 2),
('REPORT_TYPE',     'TEL_01',           '임대차 계약 해제·합의 해지 신고', 3),
-- 임대 유형
('RENTAL_TYPE',     'PRIVATE_GENERAL',  '민간일반임대',                  1),
('RENTAL_TYPE',     'PUBLIC_SUPPORT',   '공공지원민간임대',               2),
-- 단지 상태
('SITE_STATUS',     'OPERATING',        '운영중',                       1),
('SITE_STATUS',     'PRE_COMPLETION',   '준공전',                       2),
('SITE_STATUS',     'CLOSED',           '종료',                         3),
-- 보증 상태
('GUARANTEE_STATUS','APPLIED',          '신청중',                       1),
('GUARANTEE_STATUS','ISSUED',           '발급완료',                     2),
('GUARANTEE_STATUS','EXPIRED',          '만기',                         3),
('GUARANTEE_STATUS','CANCELLED',        '해지',                         4),
-- 민원 분류
('COMPLAINT_CAT',   'FACILITY',         '시설 일반',                    1),
('COMPLAINT_CAT',   'LEAK',             '누수',                         2),
('COMPLAINT_CAT',   'MOVE_IN_CHECK',    '입주 전 점검',                  3),
('COMPLAINT_CAT',   'APPLIANCE',        '가전·설비',                    4),
('COMPLAINT_CAT',   'OTHER',            '기타',                         5),
-- 조직 유형
('ORG_TYPE',        'HEAD_OFFICE',      '본사',                         1),
('ORG_TYPE',        'ASSOCIATION',      '조합',                         2),
('ORG_TYPE',        'SPC',              'SPC 법인',                     3),
('ORG_TYPE',        'AS_PARTNER',       'AS 협력업체',                  4),
('ORG_TYPE',        'REAL_ESTATE',      '공인중개사',                   5),
-- 소유권 이전 유형
('TRANSFER_TYPE',   'COMPREHENSIVE',    '포괄양수도',                   1),
('TRANSFER_TYPE',   'SALE',             '매각',                         2),
('TRANSFER_TYPE',   'INHERITANCE',      '상속',                         3);

-- ============================================================
-- SECTION 13. 트리거 — updated_at 자동 갱신
-- ============================================================

CREATE OR REPLACE FUNCTION fn_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
RETURN NEW;
END;
$$;

DO $$
DECLARE
tbl TEXT;
BEGIN
    FOREACH tbl IN ARRAY ARRAY[
        'roles','organizations','users',
        'rental_sites','rental_units',
        'tenants','lease_contracts',
        'deposit_guarantees','complaints',
        'regulatory_reports','common_codes'
    ]
    LOOP
        EXECUTE format(
            'CREATE TRIGGER trg_%s_updated_at
             BEFORE UPDATE ON %s
             FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();',
            tbl, tbl
        );
END LOOP;
END;
$$;

-- ============================================================
-- END OF SCHEMA
-- ============================================================
