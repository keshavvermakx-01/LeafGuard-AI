"""Supabase authentication and per-user persistence helpers for LeafGuard AI.
Requires: streamlit, supabase
Configure .streamlit/secrets.toml with [supabase] url, anon_key, storage_bucket.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
import streamlit as st
from supabase import Client, create_client


def get_supabase() -> Client:
    cfg = st.secrets.get("supabase", {})
    url, key = str(cfg.get("url", "")).strip(), str(cfg.get("anon_key", "")).strip()
    if not url or not key:
        st.error("Supabase is not configured. Add [supabase] url and anon_key to .streamlit/secrets.toml.")
        st.stop()
    return create_client(url, key)


def current_user() -> dict[str, Any] | None:
    return st.session_state.get("lg_user")


def is_authenticated() -> bool:
    return current_user() is not None


def render_auth_gate() -> None:
    """Show a login dialog over the public Home page."""
    if is_authenticated():
        with st.sidebar:
            st.caption(f"Signed in: {(current_user() or {}).get('email', 'user')}")
            if st.button("Log out", key="lg_logout"):
                try:
                    get_supabase().auth.sign_out()
                finally:
                    for k in ("lg_user", "lg_access_token", "lg_refresh_token"):
                        st.session_state.pop(k, None)
                st.rerun()
        return

    if st.session_state.get("lg_demo_mode"):
        st.sidebar.info("Demo mode: private account features are unavailable.")
        if st.sidebar.button("Sign in / Create account", key="lg_open_auth"):
            st.session_state["lg_auth_open"] = True
            st.rerun()
        return

    if "lg_auth_open" not in st.session_state:
        st.session_state["lg_auth_open"] = True

    if not st.session_state["lg_auth_open"]:
        if st.sidebar.button("Sign in / Create account", key="lg_reopen_auth"):
            st.session_state["lg_auth_open"] = True
            st.rerun()
        return

    @st.dialog("LeafGuard AI - Sign in or create account", width="small")
    def _auth_dialog():
        close_col, _ = st.columns([1, 8])
        with close_col:
            if st.button("Close", key="lg_close_auth", help="Close login"):
                st.session_state["lg_auth_open"] = False
                st.session_state["lg_demo_mode"] = True
                st.rerun()

        st.markdown("""
        <div style="text-align:center;margin:.2rem 0">
          <svg viewBox="0 0 64 64" width="48" height="48" aria-label="LeafGuard leaf logo">
            <path d="M53 9C31 9 13 18 12 36c0 10 7 17 17 16C47 50 54 30 53 9Z" fill="#2E8B57"/>
            <path d="M13 56C23 43 32 33 46 21" fill="none" stroke="#FFFFFF" stroke-width="3.5" stroke-linecap="round"/>
          </svg>
        </div>
        <p style="text-align:center;color:#536273">Sign in to save scans and reports privately.</p>
        """, unsafe_allow_html=True)

        client = get_supabase()
        signin, signup = st.tabs(["Sign in", "Create account"])

        with signin:
            with st.form("lg_signin_form"):
                email = st.text_input("Email", key="lg_signin_email")
                password = st.text_input("Password", type="password", key="lg_signin_password")
                submitted = st.form_submit_button("Sign in", use_container_width=True)
            if submitted:
                try:
                    result = client.auth.sign_in_with_password({"email": email.strip(), "password": password})
                    if not result.user:
                        st.error("Sign-in failed. Check your email and password.")
                    else:
                        st.session_state["lg_user"] = {"id": str(result.user.id), "email": result.user.email}
                        if result.session:
                            st.session_state["lg_access_token"] = result.session.access_token
                            st.session_state["lg_refresh_token"] = result.session.refresh_token
                        st.session_state["lg_auth_open"] = False
                        st.session_state.pop("lg_demo_mode", None)
                        st.rerun()
                except Exception:
                    st.error("Could not sign in. Check credentials and Supabase Auth settings.")

        with signup:
            with st.form("lg_signup_form"):
                name = st.text_input("Full name")
                email_new = st.text_input("Email address", key="lg_signup_email")
                password_new = st.text_input("Password (at least 8 characters)", type="password", key="lg_signup_password")
                confirm = st.text_input("Confirm password", type="password")
                submitted_new = st.form_submit_button("Create account", use_container_width=True)
            if submitted_new:
                if not name.strip():
                    st.error("Enter your name.")
                elif password_new != confirm:
                    st.error("Passwords do not match.")
                elif len(password_new) < 8:
                    st.error("Use a password with at least 8 characters.")
                else:
                    try:
                        result = client.auth.sign_up({"email": email_new.strip(), "password": password_new, "options": {"data": {"full_name": name.strip()}}})
                        if not result.user:
                            st.error("Account creation did not complete. Try again.")
                        elif not result.session:
                            st.success("Account created. Check your email to confirm it, then sign in.")
                        else:
                            st.session_state["lg_user"] = {"id": str(result.user.id), "email": result.user.email}
                            st.session_state["lg_access_token"] = result.session.access_token
                            st.session_state["lg_refresh_token"] = result.session.refresh_token
                            st.session_state["lg_auth_open"] = False
                            st.session_state.pop("lg_demo_mode", None)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Account creation failed ({type(e).__name__}): {e}")

    _auth_dialog()

def _authed_client() -> Client:
    if not current_user() or not st.session_state.get("lg_access_token"):
        raise RuntimeError("A signed-in Supabase session is required.")
    client = get_supabase()
    client.postgrest.auth(st.session_state["lg_access_token"])
    return client


def save_scan_record(*, crop: str, diagnosis: str, confidence: float | None = None, details: dict[str, Any] | None = None, report_path: str | None = None) -> dict[str, Any]:
    user = current_user()
    payload = {"user_id": user["id"], "crop": crop, "diagnosis": diagnosis, "confidence": confidence, "details": details or {}, "report_path": report_path, "created_at": datetime.now(timezone.utc).isoformat()}
    result = _authed_client().table("scan_history").insert(payload).execute()
    return result.data[0]


def list_my_scans(limit: int = 100) -> list[dict[str, Any]]:
    user = current_user()
    if not user or not st.session_state.get("lg_access_token"): return []
    result = (_authed_client().table("scan_history").select("id,created_at,crop,diagnosis,confidence,details,report_path").eq("user_id", user["id"]).order("created_at", desc=True).limit(limit).execute())
    return result.data or []


def upload_my_pdf(pdf_bytes: bytes, filename: str) -> str:
    user = current_user()
    safe = "".join(c for c in filename if c.isalnum() or c in "._-") or "report.pdf"
    path = f"{user['id']}/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{safe}"
    bucket = str(st.secrets.get("supabase", {}).get("storage_bucket", "leafguard-reports"))
    _authed_client().storage.from_(bucket).upload(path, pdf_bytes, {"content-type": "application/pdf", "upsert": "false"})
    return path


def download_my_pdf(object_path: str) -> bytes:
    user = current_user()
    if not user or not object_path.startswith(f"{user['id']}/"):
        raise PermissionError("You can only download your own reports.")
    bucket = str(st.secrets.get("supabase", {}).get("storage_bucket", "leafguard-reports"))
    return _authed_client().storage.from_(bucket).download(object_path)




