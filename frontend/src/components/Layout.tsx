import type { PropsWithChildren } from 'react';
import { NavLink } from 'react-router-dom';
import ThemeToggle from './ThemeToggle';

const navLinks = [
  { to: '/', label: 'Главная' },
  { to: '/projects', label: 'Проекты' },
  { to: '/about', label: 'Обо мне' },
  { to: '/contact', label: 'Контакты' }
];

const Layout = ({ children }: PropsWithChildren) => {
  return (
    <div className="layout">
      <header className="layout__header">
        <NavLink className="layout__logo" to="/">
          Nik52 | K3Project
        </NavLink>
        <div className="layout__controls">
          <nav className="layout__nav">
            {navLinks.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                className={({ isActive }) =>
                  isActive ? 'layout__link layout__link--active' : 'layout__link'
                }
              >
                {link.label}
              </NavLink>
            ))}
          </nav>
          <ThemeToggle />
        </div>
      </header>
      <main className="layout__main">{children}</main>
      <footer className="layout__footer">
        <p>Copyright {new Date().getFullYear()} Nik52. Все права защищены.</p>
        <p className="layout__footer-note">K3Project | Computer Vision & AI Lab</p>
      </footer>
    </div>
  );
};

export default Layout;
