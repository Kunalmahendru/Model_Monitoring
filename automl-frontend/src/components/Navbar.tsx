import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

interface NavbarProps {
  onNavigate?: (route: string) => void;
  currentPage?: string; // Add support for explicit current page
}

const Navbar: React.FC<NavbarProps> = ({ onNavigate, currentPage }) => {
  const location = useLocation();
  const navigate = useNavigate();

  const handleNavClick = (route: string) => {
    if (onNavigate) {
      onNavigate(route);
    } else {
      navigate(route);
    }
  };

  const getActiveLink = () => {
    // Use explicit currentPage if provided
    if (currentPage) {
      if (currentPage === '/deploy') return 'deploy';
      if (currentPage === '/training') return 'training';
      if (currentPage === '/live') return 'live';
      if (currentPage === '/') return 'preview';
    }
    
    // Fallback to location-based detection
    const path = location.pathname;
    if (path === '/' || path === '/homepage') return 'preview';
    if (path === '/training') return 'training';
    if (path === '/deploy') return 'deploy';
    if (path === '/livemodel') return 'live';
    return 'preview';
  };

  const isNavLinkDisabled = (targetLink: string) => {
    const current = getActiveLink();
    const flowOrder = ['preview', 'training', 'deploy', 'live'];
    const currentIndex = flowOrder.indexOf(current);
    const targetIndex = flowOrder.indexOf(targetLink);
    
    // Disable if trying to go forward in the flow
    return targetIndex > currentIndex;
  };

  const handleNavClickWithValidation = (route: string, targetLink: string) => {
    if (isNavLinkDisabled(targetLink)) {
      return; // Don't navigate if disabled
    }
    handleNavClick(route);
  };

  const activeLink = getActiveLink();

  return (
    <nav className="fixed-navbar">
      <div className="navbar-content">
        <h1 className="navbar-title">Model Monitoring</h1>
        <div className="nav-links">
          <span 
            className={`nav-link ${activeLink === 'preview' ? 'active' : ''} ${isNavLinkDisabled('preview') ? 'disabled' : ''}`}
            onClick={() => handleNavClickWithValidation('/', 'preview')}
            style={{ 
              cursor: isNavLinkDisabled('preview') ? 'not-allowed' : 'pointer',
              opacity: isNavLinkDisabled('preview') ? 0.5 : 1 
            }}
          >
            Preview
          </span>
          <span 
            className={`nav-link ${activeLink === 'training' ? 'active' : ''} ${isNavLinkDisabled('training') ? 'disabled' : ''}`}
            onClick={() => handleNavClickWithValidation('/training', 'training')}
            style={{ 
              cursor: isNavLinkDisabled('training') ? 'not-allowed' : 'pointer',
              opacity: isNavLinkDisabled('training') ? 0.5 : 1 
            }}
          >
            Training
          </span>
          <span 
            className={`nav-link ${activeLink === 'deploy' ? 'active' : ''} ${isNavLinkDisabled('deploy') ? 'disabled' : ''}`}
            onClick={() => handleNavClickWithValidation('/deploy', 'deploy')}
            style={{ 
              cursor: isNavLinkDisabled('deploy') ? 'not-allowed' : 'pointer',
              opacity: isNavLinkDisabled('deploy') ? 0.5 : 1 
            }}
          >
            Deploy
          </span>
          <span 
            className={`nav-link ${activeLink === 'live' ? 'active' : ''} ${isNavLinkDisabled('live') ? 'disabled' : ''}`}
            onClick={() => handleNavClickWithValidation('/livemodel', 'live')}
            style={{ 
              cursor: isNavLinkDisabled('live') ? 'not-allowed' : 'pointer',
              opacity: isNavLinkDisabled('live') ? 0.5 : 1 
            }}
          >
            Live
          </span>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
