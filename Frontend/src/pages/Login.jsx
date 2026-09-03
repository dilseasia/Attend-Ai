import React, { useState } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import logo from '../assets/logo.png';

const EyeIcon = ({ open }) => open ? (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" width={18} height={18}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
) : (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" width={18} height={18}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88" />
  </svg>
);

function Feature({ icon, title, desc, delay }) {
  return (
    <motion.div initial={{ opacity: 0, x: -18 }} animate={{ opacity: 1, x: 0 }} transition={{ delay, duration: 0.5 }}
      style={{ display: 'flex', alignItems: 'flex-start', gap: 14, marginBottom: 20 }}>
      <div style={{
        width: 38, height: 38, borderRadius: 10, flexShrink: 0,
        background: 'rgba(255,255,255,0.15)', border: '1px solid rgba(255,255,255,0.25)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>{icon}</div>
      <div>
        <p style={{ margin: 0, fontWeight: 700, fontSize: 13, color: '#fff' }}>{title}</p>
        <p style={{ margin: '3px 0 0', fontSize: 12, color: 'rgba(255,255,255,0.6)', lineHeight: 1.5 }}>{desc}</p>
      </div>
    </motion.div>
  );
}

export default function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showForgot, setShowForgot] = useState(false);
  const [loading, setLoading] = useState(false);
  const [focused, setFocused] = useState(null);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append('username', username);
      fd.append('password', password);
      const res = await axios.post('http://10.8.21.52:8000/api/login', fd);
      localStorage.setItem('authToken', res.data.token);
      localStorage.setItem('username', username);
      toast.success('Login successful!');
      navigate('/dashboard');
    } catch { toast.error('Invalid credentials'); }
    finally { setLoading(false); }
  };

  const inp = (f, extraPL = false) => ({
    width: '100%', boxSizing: 'border-box',
    padding: `13px 16px 13px ${extraPL ? 46 : 42}px`,
    paddingRight: f === 'password' ? 46 : 16,
    background: '#fff',
    border: focused === f ? '2px solid #7c3aed' : '1.5px solid #e2e8f0',
    borderRadius: 10, color: '#0f172a', fontSize: 14, outline: 'none',
    boxShadow: focused === f ? '0 0 0 3px rgba(124,58,237,0.12)' : '0 1px 3px rgba(0,0,0,0.06)',
    transition: 'all 0.25s',
  });

  const iconCol = (f) => ({ color: focused === f ? '#7c3aed' : '#94a3b8', transition: 'color 0.2s', display: 'flex' });

  return (
    <div style={{ minHeight: '100vh', display: 'flex', fontFamily: "'Inter','Segoe UI',system-ui,sans-serif" }}>

      {/* LEFT */}
      <motion.div initial={{ opacity: 0, x: -40 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.7, ease: [0.22,1,0.36,1] }}
        style={{
          width: '48%', minHeight: '100vh',
          background: 'linear-gradient(145deg,#6d28d9 0%,#7c3aed 35%,#9333ea 65%,#a855f7 100%)',
          padding: '48px 44px', display: 'flex', flexDirection: 'column',
          justifyContent: 'space-between', position: 'relative', overflow: 'hidden',
        }}>
        <div style={{ position:'absolute',top:-80,right:-80,width:300,height:300,borderRadius:'50%',background:'rgba(255,255,255,0.06)' }} />
        <div style={{ position:'absolute',bottom:-60,left:-60,width:260,height:260,borderRadius:'50%',background:'rgba(255,255,255,0.05)' }} />

        <div>
          {/* Brand */}
          <motion.div initial={{ opacity:0,y:-16 }} animate={{ opacity:1,y:0 }} transition={{ delay:0.2 }}
            style={{ display:'flex',alignItems:'center',gap:14,marginBottom:48 }}>
            <div style={{
              width:52,height:52,borderRadius:14,background:'rgba(255,255,255,0.95)',
              display:'flex',alignItems:'center',justifyContent:'center',
              boxShadow:'0 4px 20px rgba(0,0,0,0.15)',position:'relative',
            }}>
              <img src={logo} alt="Attend AI" style={{ width:34,height:34,objectFit:'contain' }} />
              <div style={{ position:'absolute',top:4,right:4,width:10,height:10,borderRadius:'50%',background:'#22c55e',border:'2px solid #fff' }} />
            </div>
            <div>
              <h1 style={{ margin:0,fontSize:22,fontWeight:800,color:'#fff',letterSpacing:'-0.02em' }}>Attend AI</h1>
              <p style={{ margin:0,fontSize:11,color:'rgba(255,255,255,0.7)',fontWeight:500 }}>AI Face Registration & Attendance System</p>
            </div>
          </motion.div>

          {/* Hero */}
          <motion.div initial={{ opacity:0,y:20 }} animate={{ opacity:1,y:0 }} transition={{ delay:0.3 }} style={{ marginBottom:40 }}>
            <h2 style={{ margin:'0 0 14px',fontSize:28,fontWeight:800,color:'#fff',lineHeight:1.25,letterSpacing:'-0.02em' }}>
              Intelligent Attendance Management System
            </h2>
            <p style={{ margin:0,fontSize:14,color:'rgba(255,255,255,0.7)',lineHeight:1.7,maxWidth:360 }}>
              Streamline workforce management with AI-powered attendance tracking, real-time analytics, and automated reporting.
            </p>
          </motion.div>

          {/* Features */}
          <Feature delay={0.45}
            icon={<svg width={18} height={18} fill="none" stroke="rgba(255,255,255,0.9)" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" /></svg>}
            title="AI-Powered Attendance Automation"
            desc="Reduce manual effort with intelligent face tracking"
          />
          <Feature delay={0.55}
            icon={<svg width={18} height={18} fill="none" stroke="rgba(255,255,255,0.9)" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3" /></svg>}
            title="Real-Time Analytics"
            desc="Instant insights and reporting dashboards"
          />
          <Feature delay={0.65}
            icon={<svg width={18} height={18} fill="none" stroke="rgba(255,255,255,0.9)" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" /></svg>}
            title="Enterprise-Grade Security"
            desc="End-to-end encrypted with role-based access"
          />
        </div>

        {/* Bottom status */}
        <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} transition={{ delay:0.8 }}
          style={{ borderTop:'1px solid rgba(255,255,255,0.15)',paddingTop:20,display:'flex',alignItems:'center',gap:8 }}>
          <div style={{ width:8,height:8,borderRadius:'50%',background:'#22c55e',boxShadow:'0 0 8px #22c55e' }} />
          <span style={{ fontSize:12,color:'rgba(255,255,255,0.6)',fontWeight:500 }}>System Online · All services operational</span>
        </motion.div>
      </motion.div>

      {/* RIGHT */}
      <motion.div initial={{ opacity:0,x:40 }} animate={{ opacity:1,x:0 }} transition={{ duration:0.7,ease:[0.22,1,0.36,1] }}
        style={{
          flex:1,display:'flex',flexDirection:'column',alignItems:'center',
          justifyContent:'center',padding:'40px 32px',background:'#fff',position:'relative',
        }}>
        <div style={{ width:'100%',maxWidth:380 }}>
          {/* Heading */}
          <motion.div initial={{ opacity:0,y:-12 }} animate={{ opacity:1,y:0 }} transition={{ delay:0.25 }} style={{ marginBottom:32 }}>
            <h2 style={{ margin:'0 0 6px',fontSize:26,fontWeight:800,color:'#0f172a',letterSpacing:'-0.02em' }}>Sign In</h2>
            <p style={{ margin:0,fontSize:14,color:'#94a3b8' }}>Enter your credentials to access your account</p>
          </motion.div>

          <motion.form onSubmit={handleLogin} initial={{ opacity:0 }} animate={{ opacity:1 }} transition={{ delay:0.35 }}>
            {/* Username */}
            <div style={{ marginBottom:18 }}>
              <label style={{ display:'block',fontSize:13,fontWeight:600,color:'#374151',marginBottom:7 }}>Username</label>
              <div style={{ position:'relative' }}>
                <span style={{ position:'absolute',left:13,top:'50%',transform:'translateY(-50%)', ...iconCol('username') }}>
                  <svg width={16} height={16} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
                </span>
                <input type="text" value={username} onChange={e=>setUsername(e.target.value)}
                  onFocus={()=>setFocused('username')} onBlur={()=>setFocused(null)}
                  style={inp('username')} placeholder="Enter your username" required autoComplete="username" />
              </div>
            </div>

            {/* Password */}
            <div style={{ marginBottom:10 }}>
              <label style={{ display:'block',fontSize:13,fontWeight:600,color:'#374151',marginBottom:7 }}>Password</label>
              <div style={{ position:'relative' }}>
                <span style={{ position:'absolute',left:13,top:'50%',transform:'translateY(-50%)', ...iconCol('password') }}>
                  <svg width={16} height={16} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" /></svg>
                </span>
                <input type={showPassword?'text':'password'} value={password} onChange={e=>setPassword(e.target.value)}
                  onFocus={()=>setFocused('password')} onBlur={()=>setFocused(null)}
                  style={inp('password')} placeholder="Enter your password" required autoComplete="current-password" />
                <button type="button" onClick={()=>setShowPassword(!showPassword)}
                  style={{ position:'absolute',right:13,top:'50%',transform:'translateY(-50%)',background:'none',border:'none',cursor:'pointer',color:'#94a3b8',display:'flex',padding:0 }}
                  onMouseEnter={e=>e.currentTarget.style.color='#7c3aed'}
                  onMouseLeave={e=>e.currentTarget.style.color='#94a3b8'}>
                  <EyeIcon open={showPassword} />
                </button>
              </div>
            </div>

            {/* Forgot */}
            <div style={{ textAlign:'right',marginBottom:26 }}>
              <button type="button" onClick={()=>setShowForgot(true)}
                style={{ background:'none',border:'none',cursor:'pointer',fontSize:13,fontWeight:600,color:'#7c3aed' }}
                onMouseEnter={e=>e.currentTarget.style.color='#6d28d9'}
                onMouseLeave={e=>e.currentTarget.style.color='#7c3aed'}>
                Forgot password?
              </button>
            </div>

            {/* Button */}
            <motion.button type="submit" disabled={loading}
              whileHover={!loading?{y:-2,boxShadow:'0 12px 32px rgba(124,58,237,0.4)'}:{}}
              whileTap={!loading?{scale:0.98}:{}}
              style={{
                width:'100%',padding:'14px',borderRadius:11,border:'none',
                background:loading?'linear-gradient(135deg,#c4b5fd,#a78bfa)':'linear-gradient(135deg,#7c3aed 0%,#9333ea 100%)',
                color:'#fff',fontSize:15,fontWeight:700,cursor:loading?'not-allowed':'pointer',
                boxShadow:'0 6px 20px rgba(124,58,237,0.3)',transition:'all 0.25s',
                display:'flex',alignItems:'center',justifyContent:'center',gap:8,
              }}>
              {loading ? (
                <>
                  <motion.div style={{ width:18,height:18,border:'2.5px solid rgba(255,255,255,0.3)',borderTopColor:'#fff',borderRadius:'50%' }}
                    animate={{ rotate:360 }} transition={{ duration:0.75,repeat:Infinity,ease:'linear' }} />
                  Signing in…
                </>
              ) : (
                <>
                  <svg width={16} height={16} fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M15.75 9V5.25A2.25 2.25 0 0013.5 3h-6a2.25 2.25 0 00-2.25 2.25v13.5A2.25 2.25 0 007.5 21h6a2.25 2.25 0 002.25-2.25V15m3 0l3-3m0 0l-3-3m3 3H9" /></svg>
                  Sign In
                </>
              )}
            </motion.button>
          </motion.form>

          {/* Divider */}
          <div style={{ display:'flex',alignItems:'center',gap:12,margin:'24px 0' }}>
            <div style={{ flex:1,height:1,background:'#f1f5f9' }} />
            <span style={{ fontSize:11,color:'#cbd5e1',fontWeight:500 }}>SECURE LOGIN</span>
            <div style={{ flex:1,height:1,background:'#f1f5f9' }} />
          </div>

          {/* Badge */}
          <div style={{ textAlign:'center' }}>
            <div style={{ display:'inline-flex',alignItems:'center',gap:6,background:'#f8fafc',border:'1px solid #e2e8f0',borderRadius:20,padding:'6px 14px' }}>
              <svg width={13} height={13} fill="none" stroke="#22c55e" strokeWidth={2.2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" /></svg>
              <span style={{ fontSize:12,fontWeight:600,color:'#64748b' }}>Highly Secured</span>
            </div>
          </div>
        </div>

        <div style={{ position:'absolute',bottom:20,fontSize:12,color:'#cbd5e1' }}>
          © 2025 Attend AI. All rights reserved.
        </div>
      </motion.div>

      {/* MODAL */}
      <AnimatePresence>
        {showForgot && (
          <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} exit={{ opacity:0 }}
            style={{ position:'fixed',inset:0,background:'rgba(15,23,42,0.5)',backdropFilter:'blur(8px)',display:'flex',alignItems:'center',justifyContent:'center',zIndex:100,padding:20 }}
            onClick={()=>setShowForgot(false)}>
            <motion.div initial={{ scale:0.88,opacity:0,y:20 }} animate={{ scale:1,opacity:1,y:0 }} exit={{ scale:0.88,opacity:0,y:20 }}
              transition={{ type:'spring',stiffness:320,damping:28 }} onClick={e=>e.stopPropagation()}
              style={{ background:'#fff',borderRadius:20,padding:'36px 32px',maxWidth:340,width:'100%',boxShadow:'0 32px 80px rgba(0,0,0,0.18)',textAlign:'center' }}>
              <div style={{ width:58,height:58,borderRadius:16,background:'linear-gradient(145deg,#faf5ff,#ede9fe)',border:'1.5px solid #ddd6fe',display:'flex',alignItems:'center',justifyContent:'center',margin:'0 auto 18px',fontSize:26 }}>🔑</div>
              <h3 style={{ color:'#0f172a',fontWeight:700,fontSize:18,margin:'0 0 10px' }}>Forgot Password?</h3>
              <p style={{ color:'#64748b',fontSize:13,lineHeight:1.7,margin:'0 0 24px' }}>Please contact your <strong>system administrator</strong> to reset your password.</p>
              <motion.button whileHover={{ y:-1 }} whileTap={{ scale:0.97 }} onClick={()=>setShowForgot(false)}
                style={{ padding:'11px 32px',borderRadius:10,border:'none',background:'linear-gradient(135deg,#7c3aed,#9333ea)',color:'#fff',fontWeight:600,fontSize:14,cursor:'pointer' }}>
                Got it
              </motion.button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}